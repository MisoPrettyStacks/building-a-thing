// Numerical routines and forecast-scoring statistics.
// Pure JavaScript, no dependencies. Runs unchanged in browsers and Node >= 18.

export const clip = (x, lo, hi) => (x < lo ? lo : x > hi ? hi : x);
export const sigmoid = (z) => (z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z)));
export const logit = (p) => Math.log(p / (1 - p));
export function mean(a) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i];
  return a.length ? s / a.length : NaN;
}

/* ---------- special functions ---------- */

const LANCZOS = [
  0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313,
  -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6,
  1.5056327351493116e-7,
];

/** ln Gamma(x), Lanczos approximation (g = 7, n = 9). */
export function lgamma(x) {
  if (x < 0.5) return Math.log(Math.PI / Math.abs(Math.sin(Math.PI * x))) - lgamma(1 - x);
  x -= 1;
  let a = LANCZOS[0];
  const t = x + 7.5;
  for (let i = 1; i < 9; i++) a += LANCZOS[i] / (x + i);
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
}

/** Regularised lower incomplete gamma P(a, x): series for x < a+1, continued fraction otherwise. */
function gammaP(a, x) {
  if (x <= 0) return 0;
  const gln = lgamma(a);
  if (x < a + 1) {
    let ap = a, sum = 1 / a, del = sum;
    for (let n = 0; n < 1000; n++) {
      ap += 1;
      del *= x / ap;
      sum += del;
      if (Math.abs(del) < Math.abs(sum) * 1e-16) break;
    }
    return sum * Math.exp(-x + a * Math.log(x) - gln);
  }
  const TINY = 1e-300;
  let b = x + 1 - a, c = 1 / TINY, d = 1 / b, h = d;
  for (let i = 1; i < 1000; i++) {
    const an = -i * (i - a);
    b += 2;
    d = an * d + b; if (Math.abs(d) < TINY) d = TINY;
    c = b + an / c; if (Math.abs(c) < TINY) c = TINY;
    d = 1 / d;
    const del = d * c;
    h *= del;
    if (Math.abs(del - 1) < 1e-16) break;
  }
  return 1 - Math.exp(-x + a * Math.log(x) - gln) * h;
}

export const erf = (x) => (x >= 0 ? gammaP(0.5, x * x) : -gammaP(0.5, x * x));
export const normCdf = (x) => 0.5 * (1 + erf(x / Math.SQRT2));

function betacf(a, b, x) {
  const TINY = 1e-300, EPS = 3e-14;
  const qab = a + b, qap = a + 1, qam = a - 1;
  let c = 1, d = 1 - (qab * x) / qap;
  if (Math.abs(d) < TINY) d = TINY;
  d = 1 / d;
  let h = d;
  for (let m = 1; m <= 500; m++) {
    const m2 = 2 * m;
    let aa = (m * (b - m) * x) / ((qam + m2) * (a + m2));
    d = 1 + aa * d; if (Math.abs(d) < TINY) d = TINY;
    c = 1 + aa / c; if (Math.abs(c) < TINY) c = TINY;
    d = 1 / d;
    h *= d * c;
    aa = (-(a + m) * (qab + m) * x) / ((a + m2) * (qap + m2));
    d = 1 + aa * d; if (Math.abs(d) < TINY) d = TINY;
    c = 1 + aa / c; if (Math.abs(c) < TINY) c = TINY;
    d = 1 / d;
    const del = d * c;
    h *= del;
    if (Math.abs(del - 1) < EPS) break;
  }
  return h;
}

/** Regularised incomplete beta I_x(a, b). */
export function betaInc(x, a, b) {
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  const bt = Math.exp(lgamma(a + b) - lgamma(a) - lgamma(b) + a * Math.log(x) + b * Math.log(1 - x));
  return x < (a + 1) / (a + b + 2)
    ? (bt * betacf(a, b, x)) / a
    : 1 - (bt * betacf(b, a, 1 - x)) / b;
}

/** Student-t CDF with nu degrees of freedom. */
export function tCdf(t, nu) {
  if (!isFinite(t)) return t > 0 ? 1 : 0;
  const ib = betaInc(nu / (nu + t * t), nu / 2, 0.5);
  return t > 0 ? 1 - 0.5 * ib : 0.5 * ib;
}

/** Student-t quantile by bisection on the CDF (symmetric). */
export function tQuantile(p, nu) {
  if (p === 0.5) return 0;
  if (p < 0.5) return -tQuantile(1 - p, nu);
  let lo = 0, hi = 1;
  while (tCdf(hi, nu) < p && hi < 1e12) { lo = hi; hi *= 2; }
  for (let i = 0; i < 70; i++) {
    const mid = 0.5 * (lo + hi);
    if (tCdf(mid, nu) < p) lo = mid; else hi = mid;
  }
  return 0.5 * (lo + hi);
}

/* ---------- proper scoring rules ---------- */

export const brier = (p, y) => (p - y) * (p - y);
export function logloss(p, y) {
  const q = clip(p, 1e-6, 1 - 1e-6);
  return -(y * Math.log(q) + (1 - y) * Math.log(1 - q));
}
/** Pinball (quantile) loss at level tau. */
export const pinball = (q, tau, y) => (y >= q ? tau * (y - q) : (1 - tau) * (q - y));

/* ---------- inference ---------- */

/** Deterministic PRNG (mulberry32) so every bootstrap is reproducible. */
export function mulberry32(seed) {
  let a = seed | 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Moving-block bootstrap percentile CI for the mean of a (serially dependent) series. */
export function blockBootstrapCI(x, { block = 6, B = 1000, seed = 12345, alpha = 0.05 } = {}) {
  const n = x.length;
  if (n < block * 4) return null;
  const rng = mulberry32(seed);
  const means = new Float64Array(B);
  const nb = Math.ceil(n / block);
  for (let b = 0; b < B; b++) {
    let s = 0, cnt = 0;
    for (let k = 0; k < nb; k++) {
      const st = Math.floor(rng() * (n - block + 1));
      for (let j = 0; j < block && cnt < n; j++) { s += x[st + j]; cnt++; }
    }
    means[b] = s / cnt;
  }
  means.sort();
  return [means[Math.floor((alpha / 2) * B)], means[Math.min(B - 1, Math.ceil((1 - alpha / 2) * B) - 1)]];
}

/**
 * Diebold-Mariano test for equal predictive accuracy of two forecast series.
 * la, lb: per-observation losses. Negative mean differential => forecast A is better.
 * Newey-West (Bartlett) long-run variance and the Harvey-Leybourne-Newbold small-sample correction,
 * referred to Student-t(n-1).
 */
export function dmTest(la, lb, { lag = 3, h = 3 } = {}) {
  const n = la.length;
  if (n < 30) return { n, dbar: NaN, stat: NaN, pALess: NaN, pBLess: NaN, pTwo: NaN };
  const d = new Float64Array(n);
  let dbar = 0;
  for (let i = 0; i < n; i++) { d[i] = la[i] - lb[i]; dbar += d[i]; }
  dbar /= n;
  let V = 0;
  for (let i = 0; i < n; i++) V += (d[i] - dbar) ** 2;
  V /= n;
  for (let k = 1; k <= lag; k++) {
    let g = 0;
    for (let i = k; i < n; i++) g += (d[i] - dbar) * (d[i - k] - dbar);
    V += 2 * (1 - k / (lag + 1)) * (g / n);
  }
  V = Math.max(V, 1e-20);
  const hln = Math.sqrt(Math.max(0, (n + 1 - 2 * h + (h * (h - 1)) / n) / n));
  const stat = (dbar / Math.sqrt(V / n)) * hln;
  const c = tCdf(stat, n - 1);
  return { n, dbar, stat, pALess: c, pBLess: 1 - c, pTwo: 2 * Math.min(c, 1 - c) };
}

/** Wilson score interval for a binomial proportion. */
export function wilson(k, n, z = 1.959964) {
  if (!n) return [NaN, NaN];
  const p = k / n, z2 = z * z, den = 1 + z2 / n;
  const c = (p + z2 / (2 * n)) / den;
  const m = (z * Math.sqrt((p * (1 - p)) / n + z2 / (4 * n * n))) / den;
  return [c - m, c + m];
}

/** Two-sided test of H0: P(hit)=0.5 using the normal approximation with continuity correction. */
export function binomTestHalf(k, n) {
  if (!n) return NaN;
  const z = (Math.abs(k - n / 2) - 0.5) / (Math.sqrt(n) / 2);
  return 2 * (1 - normCdf(Math.max(0, z)));
}

/**
 * Variance inflation of a mean of overlapping h-step direction outcomes under a driftless random walk:
 * corr(sign(S_t), sign(S_{t+k})) = (2/pi) asin(1 - k/h); Deff = 1 + 2 * sum_k corr_k.
 */
export function overlapDeff(h) {
  let s = 0;
  for (let k = 1; k < h; k++) s += (2 / Math.PI) * Math.asin(1 - k / h);
  return 1 + 2 * s;
}

/** Logistic recalibration y ~ sigmoid(alpha + beta * logit(p)) by Newton-Raphson; SEs from observed information. */
export function calibrationFit(ps, ys) {
  const n = ps.length;
  if (n < 50) return null;
  const x = new Float64Array(n);
  let sx = 0;
  for (let i = 0; i < n; i++) { x[i] = logit(clip(ps[i], 1e-6, 1 - 1e-6)); sx += x[i]; }
  sx /= n;
  let sv = 0;
  for (let i = 0; i < n; i++) sv += (x[i] - sx) ** 2;
  if (sv / n < 1e-10) return null;
  let a = 0, b = 1;
  let i00 = 1, i01 = 0, i11 = 1;
  for (let it = 0; it < 30; it++) {
    let g0 = 0, g1 = 0;
    i00 = 0; i01 = 0; i11 = 0;
    for (let i = 0; i < n; i++) {
      const p = sigmoid(a + b * x[i]);
      const e = ys[i] - p, w = p * (1 - p);
      g0 += e; g1 += e * x[i];
      i00 += w; i01 += w * x[i]; i11 += w * x[i] * x[i];
    }
    i00 += 1e-9; i11 += 1e-9;
    const det = i00 * i11 - i01 * i01;
    if (!(det > 1e-18)) break;
    const da = (i11 * g0 - i01 * g1) / det;
    const db = (-i01 * g0 + i00 * g1) / det;
    a += da; b += db;
    if (Math.abs(da) + Math.abs(db) < 1e-10) break;
  }
  const det = i00 * i11 - i01 * i01;
  return { alpha: a, beta: b, seAlpha: Math.sqrt(i11 / det), seBeta: Math.sqrt(i00 / det) };
}

/* ---------- aggregate score reports ---------- */

/**
 * Full report for a set of probability forecasts ps (P(up)) against outcomes ys in {0,1}.
 * Reliability bins are equal-count (quantile) bins of the forecast, because honest forecasts for a
 * near-efficient market all sit close to 0.5 and equal-width bins would be almost empty.
 */
export function binaryScores(ps, ys, { h = 3, bins = 10, boot = true } = {}) {
  const n = ps.length;
  if (!n) return { n: 0 };
  let bs = 0, ll = 0, hit = 0, calls = 0, ysum = 0;
  const L = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const p = ps[i], y = ys[i];
    L[i] = (p - y) * (p - y);
    bs += L[i];
    ll += logloss(p, y);
    ysum += y;
    if (p !== 0.5) { calls++; if (p > 0.5 === (y === 1)) hit++; }
  }
  const brierV = bs / n, ybar = ysum / n;
  const deff = overlapDeff(h);
  const nEff = Math.max(1, calls / deff);
  const hitEff = Math.round((hit / Math.max(calls, 1)) * nEff);
  const out = {
    n, brier: brierV, logloss: ll / n, ybar,
    bss50: 1 - brierV / 0.25,
    calls, hits: hit, accuracy: calls ? hit / calls : NaN,
    accCI: calls ? wilson(hitEff, nEff) : [NaN, NaN],
    accP: calls ? binomTestHalf(hitEff, Math.round(nEff)) : NaN,
    nEff: calls / deff, deff,
    brierCI: boot ? blockBootstrapCI(L, { block: 12 }) : null,
  };
  // equal-count reliability bins
  const nb = Math.max(1, Math.min(bins, Math.floor(n / 20)));
  const idx = Array.from({ length: n }, (_, i) => i).sort((a, b) => ps[a] - ps[b]);
  const B = [];
  for (let b = 0; b < nb; b++) {
    const lo = Math.floor((b * n) / nb), hi = Math.floor(((b + 1) * n) / nb);
    let sp = 0, sy = 0;
    for (let k = lo; k < hi; k++) { sp += ps[idx[k]]; sy += ys[idx[k]]; }
    const c = hi - lo;
    if (c) B.push({ n: c, p: sp / c, y: sy / c });
  }
  let ece = 0, rel = 0, res = 0;
  for (const b of B) {
    ece += (b.n / n) * Math.abs(b.y - b.p);
    rel += (b.n / n) * (b.p - b.y) ** 2;
    res += (b.n / n) * (b.y - ybar) ** 2;
  }
  out.bins = B; out.ece = ece; out.rel = rel; out.res = res; out.unc = ybar * (1 - ybar);
  out.calib = calibrationFit(ps, ys);
  return out;
}

/** Interval coverage and pinball loss for quantile forecasts. recs: [{q:[...7 values], r}] in log-return units. */
export function quantileScores(recs, levels) {
  const n = recs.length;
  if (!n) return { n: 0 };
  const iq = (v) => levels.indexOf(v);
  const pairs = { c90: [iq(0.05), iq(0.95)], c80: [iq(0.1), iq(0.9)], c50: [iq(0.25), iq(0.75)] };
  const out = { n };
  for (const [k, [a, b]] of Object.entries(pairs)) {
    let c = 0;
    for (const r of recs) if (r.r >= r.q[a] && r.r <= r.q[b]) c++;
    out[k] = c / n;
  }
  let pl = 0;
  for (const r of recs) {
    let s = 0;
    for (let j = 0; j < levels.length; j++) s += pinball(r.q[j], levels[j], r.r);
    pl += s / levels.length;
  }
  out.pinballBps = (pl / n) * 1e4;
  return out;
}
