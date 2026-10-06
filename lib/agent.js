// The accuracy-guardian agent. It is an autonomous, fully auditable statistical agent (no paid LLM, no API keys):
//   observe  -> score the live ledger against baselines
//   hypothesise -> propose small perturbations of the champion configuration (champion/challenger)
//   test     -> walk-forward on held-out recent bars, paired Diebold-Mariano test with Bonferroni correction
//   act      -> adopt only if the improvement is large, significant and stable; otherwise keep the champion
//   verify   -> replay the previous champion over the live period and roll back if it would have done better
//   guard    -> data-driven shrinkage toward 50% when the live record does not support the model's confidence
// Every decision is written to agent_log.jsonl with the numbers that justified it.

import { walkForward, ALL_FEATURES, DEFAULT_CONFIG } from './engine.js';
import { brier, dmTest, mean, mulberry32 } from './stats.js';

const HOUR = 3600;
export const AGENT_PARAMS = {
  deepEverySec: 6 * HOUR,      // scheduled challenger search
  earlyEverySec: 1 * HOUR,     // minimum gap when degradation triggers an early search
  minAdoptGapSec: 24 * HOUR,
  candidates: 40,
  testBars: 1440,              // 5 days held-out block (5-min bars)
  minEffect: 2e-4,             // minimum Brier improvement over the champion
  alpha: 0.05,                 // family-wise error rate, Bonferroni-split over candidates
  shrinkMinN: 576,             // 2 days of live forecasts before shrinkage is revisited
  shrinkPrior: 0.2,            // ridge pseudo-evidence pulling the optimal shrink toward 0
  shrinkHysteresis: 0.1,
  rollbackMinN: 288,
  rollbackMaxAgeSec: 7 * 86400,
};

const TUNE = {
  volLambda: { lo: 0.85, hi: 0.985, kind: 'decay' },
  driftLambda: { lo: 0.99, hi: 0.9998, kind: 'decay' },
  baseLambda: { lo: 0.99, hi: 0.9998, kind: 'decay' },
  kLambda: { lo: 0.98, hi: 0.999, kind: 'decay' },
  kurtLambda: { lo: 0.99, hi: 0.9995, kind: 'decay' },
  lrLogit: { lo: 0.005, hi: 0.3, kind: 'log' },
  l2Logit: { lo: 1e-5, hi: 0.1, kind: 'log' },
  lrShort: { lo: 0.005, hi: 0.3, kind: 'log' },
  l2Short: { lo: 1e-5, hi: 0.1, kind: 'log' },
  hedgeEta: { lo: 0.5, hi: 40, kind: 'log' },
  fixedShare: { lo: 0.001, hi: 0.1, kind: 'log' },
  plattLr: { lo: 1e-4, hi: 0.02, kind: 'log' },
};

export function proposeCandidates(champion, rng, K) {
  const names = Object.keys(TUNE);
  const out = [];
  for (let c = 0; c < K; c++) {
    const cand = { ...champion, features: champion.features.slice() };
    const desc = [];
    if (rng() < 0.3) {
      const f = ALL_FEATURES[Math.floor(rng() * ALL_FEATURES.length)];
      const has = cand.features.includes(f);
      if (has && cand.features.length > 3) { cand.features = cand.features.filter((x) => x !== f); desc.push('-' + f); }
      else if (!has) { cand.features = ALL_FEATURES.filter((x) => x === f || cand.features.includes(x)); desc.push('+' + f); }
    } else {
      const m = 1 + Math.floor(rng() * 3);
      const pool = names.slice();
      for (let k = 0; k < m; k++) {
        const nm = pool.splice(Math.floor(rng() * pool.length), 1)[0];
        const sp = TUNE[nm], f = Math.exp((rng() * 2 - 1) * Math.LN2);
        const old = cand[nm];
        let v = sp.kind === 'decay' ? 1 - (1 - old) * f : old * f;
        v = Math.min(sp.hi, Math.max(sp.lo, v));
        cand[nm] = v;
        desc.push(`${nm}:${old.toPrecision(3)}->${v.toPrecision(3)}`);
      }
    }
    if (desc.length) out.push({ cfg: cand, desc: desc.join(', ') });
  }
  return out;
}

/** Walk-forward losses on the PRE-shrink probability (so model quality and the confidence guard stay independent). */
function lossesOn(bars, cfg, startIdx) {
  const steps = walkForward(bars, cfg).steps.filter((s) => s.i >= startIdx && s.y !== null);
  return { idx: steps.map((s) => s.i), loss: steps.map((s) => brier(s.praw, s.y)) };
}

/** Champion/challenger search. Returns the decision and the evidence. */
export function searchChallenger({ bars, champion, seed, K = AGENT_PARAMS.candidates, testBars = AGENT_PARAMS.testBars }) {
  const n = bars.length;
  const startIdx = n - testBars;
  if (startIdx < champion.minHist + 1500) return { decision: 'skip', reason: 'not enough history for a held-out block' };
  const rng = mulberry32(seed);
  const base = lossesOn(bars, champion, startIdx);
  const cands = proposeCandidates(champion, rng, K).map((c) => ({ ...c, ...lossesOn(bars, c.cfg, startIdx) }));
  const m0 = mean(base.loss);
  const ranked = cands
    .map((c) => ({ ...c, brier: mean(c.loss), delta: m0 - mean(c.loss) }))
    .sort((a, b) => b.delta - a.delta);
  const best = ranked[0];
  const evidence = {
    championBrier: m0, n: base.loss.length, tested: cands.length, seed,
    top: ranked.slice(0, 3).map((c) => ({ change: c.desc, brier: c.brier, delta: c.delta })),
  };
  if (!best) return { decision: 'keep', reason: 'no valid candidates', evidence };
  const dm = dmTest(best.loss, base.loss, { h: champion.h });
  const alphaCorr = AGENT_PARAMS.alpha / cands.length;
  const half = Math.floor(base.loss.length / 2);
  const d1 = mean(base.loss.slice(0, half)) - mean(best.loss.slice(0, half));
  const d2 = mean(base.loss.slice(half)) - mean(best.loss.slice(half));
  evidence.best = { change: best.desc, brier: best.brier, delta: best.delta, dmP: dm.pALess, alphaCorrected: alphaCorr, halves: [d1, d2] };
  const pass = best.delta >= AGENT_PARAMS.minEffect && dm.pALess < alphaCorr && d1 > 0 && d2 > 0;
  if (!pass) {
    const why = [];
    if (best.delta < AGENT_PARAMS.minEffect) why.push(`effect ${best.delta.toExponential(2)} < ${AGENT_PARAMS.minEffect}`);
    if (!(dm.pALess < alphaCorr)) why.push(`DM p=${dm.pALess.toFixed(4)} >= Bonferroni ${alphaCorr.toExponential(2)}`);
    if (!(d1 > 0 && d2 > 0)) why.push('improvement not stable across both halves');
    return { decision: 'keep', reason: why.join('; '), evidence };
  }
  return { decision: 'adopt', candidate: best.cfg, change: best.desc, evidence };
}

/** Ridge-regularised optimal shrinkage: argmin_lambda mean (0.5 + lambda (praw - 0.5) - y)^2. */
export function optimalShrink(resolved) {
  let num = 0, den = 0;
  for (const f of resolved) {
    const d = f.p_raw - 0.5;
    num += d * (f.res.y - 0.5);
    den += d * d;
  }
  const lam = num / (den + AGENT_PARAMS.shrinkPrior);
  return { lambda: Math.min(1, Math.max(0, lam)), rawLambda: num / (den || 1), n: resolved.length };
}

/** Would the previous champion, replayed over the live period, have beaten what was actually issued? */
export function rollbackCheck({ bars, prevCfg, resolved, sinceTs, version }) {
  const live = resolved.filter((f) => f.t_issue >= sinceTs && f.cfg_version >= version);
  if (live.length < AGENT_PARAMS.rollbackMinN) return { enough: false, n: live.length };
  const replay = new Map();
  for (const s of walkForward(bars, prevCfg).steps) if (s.y !== null) replay.set(s.t + 300, s);
  const A = [], B = [];
  for (const f of live) {
    const s = replay.get(f.t_issue);
    if (!s) continue;
    A.push(brier(f.p_raw, f.res.y)); B.push(brier(s.praw, s.y));
  }
  if (A.length < AGENT_PARAMS.rollbackMinN) return { enough: false, n: A.length };
  const dm = dmTest(A, B, { h: prevCfg.h });
  return { enough: true, n: A.length, liveBrier: mean(A), prevBrier: mean(B), dbar: dm.dbar, pPrevBetter: dm.pBLess,
    rollback: dm.dbar > 1e-4 && dm.pBLess < 0.05 };
}

/** Simple degradation alarm on the most recent day of live forecasts. */
export function degradationAlarm(resolved) {
  const w = resolved.slice(-288);
  if (w.length < 288) return { alarm: false };
  const a = w.map((f) => brier(f.p, f.res.y)), b = w.map((f) => brier(f.m[0], f.res.y));
  const dm = dmTest(a, b, { h: 3 });
  return { alarm: dm.dbar > 5e-4 && dm.pBLess < 0.1, dbar: dm.dbar, p: dm.pBLess };
}

export function freshAgentState() {
  return { lastDeepTs: 0, lastAdoptTs: 0, adoptedAtTs: 0, lastShrinkTs: 0, version: 1 };
}

/**
 * One agent review. Pure with respect to I/O: returns the (possibly new) config document and the events to log.
 * `config` = { champion, previous, history, agent }.
 */
export function runAgent({ nowSec, resolved, bars, config }) {
  const events = [];
  let { champion, previous, history = [], agent = freshAgentState() } = config;
  agent = { ...freshAgentState(), ...agent };
  const bump = (patch, kind, detail) => {
    previous = champion;
    champion = { ...champion, ...patch, version: champion.version + 1 };
    history.push({ ts: nowSec, version: champion.version, kind, detail });
    history = history.slice(-200);
  };

  // 1) shrinkage guard (cheap, no refit; p_raw is independent of shrink so past evidence stays valid)
  if (resolved.length >= AGENT_PARAMS.shrinkMinN && nowSec - agent.lastShrinkTs >= 24 * HOUR) {
    const s = optimalShrink(resolved.slice(-4032));
    agent.lastShrinkTs = nowSec;
    const target = Math.round(s.lambda * 20) / 20;
    if (Math.abs(target - champion.shrink) >= AGENT_PARAMS.shrinkHysteresis) {
      const from = champion.shrink;
      const keepPrev = previous;
      bump({ shrink: target }, 'shrink', { from, to: target, evidence: s });
      previous = keepPrev; // shrink changes never replace the rollback target
      events.push({ type: 'shrink', from, to: target, evidence: s });
    } else {
      events.push({ type: 'shrink-review', current: champion.shrink, suggested: s.lambda, n: s.n, action: 'none (inside hysteresis band)' });
    }
  }

  // 2) deep cycle: rollback check + challenger search
  const alarm = degradationAlarm(resolved);
  const due = nowSec - agent.lastDeepTs >= AGENT_PARAMS.deepEverySec;
  const early = alarm.alarm && nowSec - agent.lastDeepTs >= AGENT_PARAMS.earlyEverySec;
  if (due || early) {
    agent.lastDeepTs = nowSec;
    let rolledBack = false;
    if (agent.adoptedAtTs && previous && nowSec - agent.adoptedAtTs <= AGENT_PARAMS.rollbackMaxAgeSec) {
      const rb = rollbackCheck({ bars, prevCfg: { ...previous, shrink: champion.shrink }, resolved, sinceTs: agent.adoptedAtTs, version: champion.version });
      events.push({ type: 'rollback-check', ...rb });
      if (rb.enough && rb.rollback) {
        const bad = champion.version;
        champion = { ...previous, shrink: champion.shrink, version: champion.version + 1 };
        previous = null;
        agent.adoptedAtTs = 0;
        history.push({ ts: nowSec, version: champion.version, kind: 'rollback', detail: { from: bad, evidence: rb } });
        events.push({ type: 'rollback', from_version: bad, to_version: champion.version });
        rolledBack = true;
      }
    }
    if (!rolledBack && nowSec - agent.lastAdoptTs >= AGENT_PARAMS.minAdoptGapSec) {
      const r = searchChallenger({ bars, champion, seed: nowSec });
      if (r.decision === 'adopt') {
        const old = champion;
        champion = { ...r.candidate, shrink: old.shrink, version: old.version + 1 };
        previous = old;
        agent.lastAdoptTs = nowSec; agent.adoptedAtTs = nowSec;
        history.push({ ts: nowSec, version: champion.version, kind: 'adopt', detail: { change: r.change, evidence: r.evidence } });
        events.push({ type: 'adopt', version: champion.version, change: r.change, evidence: r.evidence });
      } else {
        events.push({ type: 'search', decision: r.decision, reason: r.reason, evidence: r.evidence });
      }
    } else if (!rolledBack) {
      events.push({ type: 'search', decision: 'skip', reason: 'adoption cooldown (24h) active' });
    }
    if (alarm.alarm) events.push({ type: 'alarm', ...alarm });
  }
  return { config: { champion, previous, history, agent }, events };
}

export const INITIAL_CONFIG = () => ({ champion: { ...DEFAULT_CONFIG, features: DEFAULT_CONFIG.features.slice() }, previous: null, history: [], agent: freshAgentState() });
