// Builds the public scoreboard (summary.json) from the append-only ledger.
// Isomorphic: only uses lib/stats.js, so the same numbers can be recomputed by anyone from the ledger.
import { binaryScores, quantileScores, dmTest, mean, brier } from './stats.js';
import { QLEVELS, MEMBER_NAMES, LADDER_BPS } from './engine.js';

/** Join forecast events with their resolution events. */
export function joinLedger(records) {
  const byId = new Map();
  let gaps = 0, voids = 0;
  for (const r of records) {
    if (r.type === 'forecast') byId.set(r.id, { ...r, res: null, void: null });
    else if (r.type === 'resolution' && byId.has(r.id)) byId.get(r.id).res = r;
    else if (r.type === 'void' && byId.has(r.id)) { byId.get(r.id).void = r; voids++; }
    else if (r.type === 'gap') gaps++;
  }
  const all = [...byId.values()].sort((a, b) => a.t_issue - b.t_issue);
  const resolved = all.filter((f) => f.res && f.res.y !== null && f.res.y !== undefined);
  const ties = all.filter((f) => f.res && (f.res.y === null || f.res.y === undefined)).length;
  const pending = all.filter((f) => !f.res && !f.void);
  return { all, resolved, pending, ties, gaps, voids };
}

function windowReport(rs, h) {
  if (!rs.length) return { n: 0 };
  const ps = rs.map((f) => f.p), ys = rs.map((f) => f.res.y);
  const main = binaryScores(ps, ys, { h });
  const lossModel = ps.map((p, i) => brier(p, ys[i]));
  const lossHalf = ys.map(() => 0.25);
  const lossBase = rs.map((f, i) => brier(f.m[0], ys[i]));
  const members = {};
  MEMBER_NAMES.forEach((nm, k) => { members[nm] = mean(rs.map((f, i) => brier(f.m[k], ys[i]))); });
  const q = rs.filter((f) => f.q && f.res.r !== null && f.res.r !== undefined).map((f) => ({ q: f.q, r: f.res.r }));
  // threshold-contract (strike ladder) Brier: P(S > x_j) for fixed bps offsets, vs. per-strike climatology
  const lad = rs.filter((f) => f.ladder && f.res.r !== null && f.res.r !== undefined);
  let ladder = null;
  if (lad.length >= 30) {
    const per = LADDER_BPS.map((b, j) => {
      const ys2 = lad.map((f) => (f.res.r > b / 1e4 ? 1 : 0)), yb = mean(ys2);
      return { bps: b, brier: mean(lad.map((f, i) => brier(f.ladder[j], ys2[i]))), climatology: yb * (1 - yb) };
    });
    ladder = { n: lad.length, brier: mean(per.map((x) => x.brier)), climatology: mean(per.map((x) => x.climatology)), per };
  }
  return {
    ...main, ladder,
    brierBase: mean(lossBase),
    bssBase: 1 - main.brier / mean(lossBase),
    dmVs50: dmTest(lossModel, lossHalf, { h }),
    dmVsBase: dmTest(lossModel, lossBase, { h }),
    members,
    interval: quantileScores(q, QLEVELS),
    first: rs[0].t_issue, last: rs[rs.length - 1].t_issue,
  };
}

export function rollingSeries(resolved, win = 144, every = 12, maxPts = 300) {
  const out = [];
  if (resolved.length < win) return out;
  const lm = resolved.map((f) => brier(f.p, f.res.y));
  const lb = resolved.map((f) => brier(f.m[0], f.res.y));
  let sm = 0, sb = 0;
  for (let i = 0; i < resolved.length; i++) {
    sm += lm[i]; sb += lb[i];
    if (i >= win) { sm -= lm[i - win]; sb -= lb[i - win]; }
    if (i >= win - 1 && (i - (win - 1)) % every === 0) out.push({ t: resolved[i].t_issue, model: sm / win, base: sb / win });
  }
  return out.slice(-maxPts);
}

export function buildSummary({ records, config, agent, extras = {}, nowSec = Math.floor(Date.now() / 1000) }) {
  const h = config.champion.h;
  const { all, resolved, pending, ties, gaps, voids } = joinLedger(records);
  const day = 86400;
  const last24 = resolved.filter((f) => f.t_issue >= nowSec - day);
  const last7 = resolved.filter((f) => f.t_issue >= nowSec - 7 * day);
  const latest = all.length ? all[all.length - 1] : null;
  const slim = (f) => f && {
    id: f.id, t_issue: f.t_issue, target_t: f.target_t, p: f.p, p_raw: f.p_raw, c0: f.c0, q: f.q, nu: f.nu,
    m: f.m, ladder: f.ladder, cfg_version: f.cfg_version, cfg_hash: f.cfg_hash, source: f.source, hash: f.hash,
    res: f.res && { c1: f.res.c1, y: f.res.y, r: f.res.r },
  };
  return {
    generated_at: new Date(nowSec * 1000).toISOString(),
    horizon_minutes: (h * 300) / 60,
    counts: { forecasts: all.length, resolved: resolved.length, pending: pending.length, ties, gaps, voids },
    latest: slim(latest),
    recent: resolved.slice(-60).map((f) => ({ t: f.t_issue, p: f.p, y: f.res.y, c0: f.c0, c1: f.res.c1 })),
    windows: { all: windowReport(resolved, h), last7d: windowReport(last7, h), last24h: windowReport(last24, h) },
    rolling: rollingSeries(resolved),
    config: { version: config.champion.version, params: config.champion },
    agent,
    ...extras,
  };
}
