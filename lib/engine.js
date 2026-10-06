// Forecasting engine: causal features, four base models, log-odds pooling with fixed-share Hedge,
// online Platt recalibration, shrinkage, and a coherent Student-t predictive distribution.
//
// One code path serves live forecasting, the agent's challenger search and the in-browser backtest:
// walkForward() replays bars in time order; at step i it may only use bars <= i, and the label of step
// j = i - h (known only once bar i has closed) is the only feedback it receives. See scripts/selftest.js
// for the leakage test that enforces this.

import { clip, sigmoid, logit, normCdf, tQuantile, tCdf, logloss } from './stats.js';

export const STEP = 300; // seconds per bar
export const QLEVELS = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95];
// Fixed strike ladder, in basis points of log-return relative to the price at issue (threshold-contract scoring).
export const LADDER_BPS = [-40, -25, -10, 10, 25, 40];
export const ALL_FEATURES = ['z1', 'z3', 'z12', 'z36', 'z144', 'z288', 'dev12', 'dev48', 'volsign'];
export const MEMBER_NAMES = ['base', 'logit', 'drift', 'short'];

export const DEFAULT_CONFIG = {
  version: 1,
  h: 3,                      // forecast horizon in bars (3 x 5 min = 15 min)
  minHist: 300,              // bars of history before the first forecast
  volLambda: 0.94,           // EWMA variance decay (RiskMetrics)
  driftLambda: 0.999,
  baseLambda: 0.999,
  lrLogit: 0.05, l2Logit: 0.001,
  lrShort: 0.05, l2Short: 0.001,
  hedgeEta: 5, fixedShare: 0.01,
  plattLr: 0.002, plattAMax: 1.5,
  shrink: 1.0,
  pMin: 0.02, pMax: 0.98,
  kLambda: 0.995,            // EWMA decay for realised/predicted variance ratio and kurtosis
  kurtLambda: 0.998,
  features: ALL_FEATURES.slice(),
};

/** Fill gaps on the regular grid by carrying the previous close forward (zero volume). */
export function gridBars(raw, step = STEP, endT = 0) {
  if (!raw.length) return [];
  const out = [raw[0]];
  let idx = 1, prev = raw[0];
  const end = Math.max(raw[raw.length - 1].t, endT);
  for (let t = raw[0].t + step; t <= end; t += step) {
    while (idx < raw.length && raw[idx].t < t) idx++;
    if (idx < raw.length && raw[idx].t === t) { prev = raw[idx]; out.push(prev); idx++; }
    else out.push({ t, o: prev.c, h: prev.c, l: prev.c, c: prev.c, v: 0, filled: true });
  }
  return out;
}

const cap = (x, m) => (x > m ? m : x < -m ? -m : x);

/** Causal feature computation. Everything at index i uses bars 0..i only. */
function computeFeatures(bars, cfg) {
  const n = bars.length;
  const lc = new Float64Array(n);
  for (let i = 0; i < n; i++) lc[i] = Math.log(bars[i].c);
  const sig = new Float64Array(n), mu = new Float64Array(n);
  const X = new Array(n).fill(null), X3 = new Array(n).fill(null);
  const a12 = 2 / 13, a48 = 2 / 49;
  let ev = 1e-8, e12 = lc[0], e48 = lc[0], muv = 0, sm = null, sv = 1;
  const lags = { z1: 1, z3: 3, z12: 12, z36: 36, z144: 144, z288: 288 };
  for (let i = 0; i < n; i++) {
    const r = i ? lc[i] - lc[i - 1] : 0;
    if (i === 1) ev = r * r || 1e-8;
    else if (i > 1) ev = cfg.volLambda * ev + (1 - cfg.volLambda) * r * r;
    sig[i] = Math.max(Math.sqrt(ev), 1e-5);
    e12 += a12 * (lc[i] - e12);
    e48 += a48 * (lc[i] - e48);
    muv = cfg.driftLambda * muv + (1 - cfg.driftLambda) * r;
    mu[i] = muv;
    const s = Math.log1p(bars[i].v);
    let zvol = 0;
    if (sm === null) sm = s;
    else {
      zvol = (s - sm) / Math.sqrt(Math.max(sv, 1e-6));
      const d = s - sm;
      sm += 0.01 * d;
      sv = 0.99 * (sv + 0.01 * d * d);
    }
    if (i < cfg.minHist) continue;
    const f = {};
    for (const k in lags) f[k] = cap((lc[i] - lc[i - lags[k]]) / (sig[i] * Math.sqrt(lags[k])), 6);
    f.dev12 = cap((lc[i] - e12) / (sig[i] * Math.sqrt(12)), 6);
    f.dev48 = cap((lc[i] - e48) / (sig[i] * Math.sqrt(48)), 6);
    f.volsign = Math.sign(r) * cap(zvol, 4);
    X[i] = cfg.features.map((k) => f[k]).concat(1);
    X3[i] = [f.z1, f.z3, 1];
  }
  return { lc, sig, mu, X, X3 };
}

function adagrad(w, G, x, err, lr, l2) {
  const d = w.length;
  for (let k = 0; k < d; k++) {
    const g = err * x[k] + (k < d - 1 ? l2 * w[k] : 0);
    G[k] += g * g;
    w[k] -= (lr * g) / (Math.sqrt(G[k]) + 1e-8);
  }
}
const dot = (w, x) => { let s = 0; for (let k = 0; k < w.length; k++) s += w[k] * x[k]; return s; };

const tqCache = new Map();
function tLevels(nu) {
  const key = Math.round(nu * 2) / 2;
  let v = tqCache.get(key);
  if (!v) { v = QLEVELS.map((l) => tQuantile(l, key)); tqCache.set(key, v); }
  return [key, v];
}

/**
 * Replay bars in time order.
 * opts.quantFrom: compute predictive quantiles for steps with i >= quantFrom (default: none).
 * Returns { steps, state }. A step is { i, t, p, praw, m, z, y, r, c0, c1, q?, nu?, scale? } where
 * t is the START time of bar i (the forecast is issued at t + STEP, when bar i closes).
 */
export function walkForward(bars, cfg, opts = {}) {
  const n = bars.length, h = cfg.h;
  const { lc, sig, mu, X, X3 } = computeFeatures(bars, cfg);
  const d1 = cfg.features.length + 1;
  const w1 = new Float64Array(d1), G1 = new Float64Array(d1);
  const w3 = new Float64Array(3), G3 = new Float64Array(3);
  const K = 4;
  let ew = new Float64Array(K).fill(1 / K);
  let a = 1, b = 0;
  let base = 0.5;
  let m2 = 1, m4 = 6; // EWMA of u^2 and u^4, u = realised h-step return / predicted sd
  const steps = new Array(n).fill(null);
  const quantFrom = opts.quantFrom ?? Infinity;
  const pLo = cfg.pMin, pHi = cfg.pMax;

  for (let i = cfg.minHist; i < n; i++) {
    // 1) feedback for step j = i - h, whose outcome is now known (uses c[i], which has closed)
    const j = i - h;
    if (j >= cfg.minHist) {
      const sj = steps[j];
      const dir = lc[i] - lc[j];
      const u = dir / (sig[j] * Math.sqrt(h));
      m2 = cfg.kLambda * m2 + (1 - cfg.kLambda) * u * u;
      m4 = cfg.kurtLambda * m4 + (1 - cfg.kurtLambda) * u ** 4;
      if (dir !== 0) {
        const y = dir > 0 ? 1 : 0;
        const p1 = sigmoid(dot(w1, X[j]));
        adagrad(w1, G1, X[j], p1 - y, cfg.lrLogit, cfg.l2Logit);
        const p3 = sigmoid(dot(w3, X3[j]));
        adagrad(w3, G3, X3[j], p3 - y, cfg.lrShort, cfg.l2Short);
        base = cfg.baseLambda * base + (1 - cfg.baseLambda) * y;
        // fixed-share Hedge on the stored member forecasts
        let tot = 0;
        for (let k = 0; k < K; k++) { ew[k] *= Math.exp(-cfg.hedgeEta * logloss(sj.m[k], y)); tot += ew[k]; }
        for (let k = 0; k < K; k++) ew[k] = (1 - cfg.fixedShare) * (ew[k] / tot) + cfg.fixedShare / K;
        // online Platt scaling on the stored pooled logit
        const e = sigmoid(a * sj.z + b) - y;
        a = clip(a - cfg.plattLr * e * sj.z, 0, cfg.plattAMax);
        b = clip(b - cfg.plattLr * 0.25 * e, -0.2, 0.2);
      }
    }
    // 2) forecast at i
    const x = X[i];
    const pm = [
      clip(base, pLo, pHi),
      clip(sigmoid(dot(w1, x)), pLo, pHi),
      clip(normCdf((mu[i] * Math.sqrt(h)) / sig[i]), pLo, pHi),
      clip(sigmoid(dot(w3, X3[i])), pLo, pHi),
    ];
    let z = 0;
    for (let k = 0; k < K; k++) z += ew[k] * logit(pm[k]);
    const praw = sigmoid(a * z + b);
    const p = clip(0.5 + cfg.shrink * (praw - 0.5), pLo, pHi);
    const c0 = bars[i].c;
    const c1 = i + h < n ? bars[i + h].c : null;
    const st = {
      i, t: bars[i].t, p, praw, m: pm, z, c0, c1,
      y: c1 === null || c1 === c0 ? null : c1 > c0 ? 1 : 0,
      r: c1 === null ? null : Math.log(c1 / c0),
    };
    if (i >= quantFrom) {
      const kappa = m4 / (m2 * m2);
      const nuRaw = kappa > 3.3 ? 4 + 6 / (kappa - 3) : 30;
      const [nu, tl] = tLevels(clip(nuRaw, 3.5, 30));
      const scale = (Math.sqrt(m2) * sig[i] * Math.sqrt(h)) / Math.sqrt(nu / (nu - 2));
      const shift = tQuantile(p, nu);
      st.q = tl.map((v) => scale * (shift + v));
      st.nu = nu; st.scale = scale;
      // P(S > x) = T_nu(mu' - x/scale), coherent with p (x = 0 gives p)
      st.ladder = LADDER_BPS.map((b) => clip(tCdf(shift - b / 1e4 / scale, nu), 0.001, 0.999));
    }
    steps[i] = st;
  }
  const out = steps.filter(Boolean);
  const state = {
    features: cfg.features.slice(), w1: Array.from(w1), w3: Array.from(w3),
    weights: Array.from(ew), a, b, base, m2, m4,
  };
  return { steps: out, state };
}

/** Forecast issued when the last bar of `bars` has just closed. */
export function forecastLatest(bars, cfg) {
  const { steps, state } = walkForward(bars, cfg, { quantFrom: bars.length - 1 });
  const last = steps[steps.length - 1];
  if (!last || last.i !== bars.length - 1) throw new Error('not enough history for a forecast');
  return { step: last, state };
}
