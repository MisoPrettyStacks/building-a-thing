// Plumbing test: ledger -> join -> summary -> agent, driven by a deterministic fixture (NOT data; test only).
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { forecastLatest, walkForward, DEFAULT_CONFIG, QLEVELS } from '../lib/engine.js';
import { appendRecord, readLedger, verifyChain } from '../lib/io.js';
import { buildSummary, joinLedger } from '../lib/summary.js';
import { runAgent, INITIAL_CONFIG, optimalShrink, searchChallenger } from '../lib/agent.js';
import { mulberry32, mean } from '../lib/stats.js';

const rng = mulberry32(2024);
const gauss = () => Math.sqrt(-2 * Math.log(1 - rng())) * Math.cos(2 * Math.PI * rng());
let lp = Math.log(0.5);
const bars = [];
for (let i = 0; i < 4400; i++) {
  const o = Math.exp(lp); lp += 0.0012 * gauss(); const c = Math.exp(lp);
  bars.push({ t: 1.7e9 + i * 300, o, h: Math.max(o, c), l: Math.min(o, c), c, v: 1000 + 300 * rng() });
}

const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'pipe-'));
const cfg = INITIAL_CONFIG();
const h = 3;
const start = 3700, end = bars.length - 1;
const t0 = Date.now();
for (let i = start; i <= end; i++) {
  const t = bars[i].t + 300;
  const iso = new Date(t * 1000).toISOString();
  // resolve forecast made h bars ago
  if (i - h >= start) {
    const f = bars[i - h];
    appendRecord(dir, { type: 'resolution', id: f.t, c1: bars[i].c, y: bars[i].c > f.c ? 1 : bars[i].c < f.c ? 0 : null, r: Math.log(bars[i].c / f.c) }, iso);
  }
  const { step } = forecastLatest(bars.slice(0, i + 1), cfg.champion);
  appendRecord(dir, {
    type: 'forecast', id: bars[i].t, bar_t: bars[i].t, t_issue: t, target_t: t + h * 300, p: step.p, p_raw: step.praw,
    m: step.m, q: step.q, q_levels: QLEVELS, nu: step.nu, c0: step.c0, cfg_version: 1, cfg_hash: 'x', source: 'fixture',
  }, iso);
}
console.log(`built ledger of ${readLedger(dir).length} events in ${((Date.now() - t0) / 1000).toFixed(1)}s`);
assert.ok(verifyChain(dir).ok);

const records = readLedger(dir);
const J = joinLedger(records);
assert.equal(J.pending.length, h);
assert.ok(J.resolved.length > 600);

const nowSec = bars[end].t + 300;
const out = runAgent({ nowSec, resolved: J.resolved, bars, config: cfg });
for (const e of out.events) console.log('agent event:', e.type, e.decision || e.action || '', e.reason || '');
assert.ok(out.events.some((e) => e.type === 'search' || e.type === 'adopt'), 'agent ran a search');
const s = buildSummary({ records, config: out.config, agent: { events: out.events }, nowSec });
const w = s.windows.all;
console.log(`summary: n=${w.n} brier=${w.brier.toFixed(5)} bss50=${(100 * w.bss50).toFixed(3)}% ece=${w.ece.toFixed(4)} cov80=${w.interval.c80.toFixed(3)} cov50=${w.interval.c50.toFixed(3)}`);
assert.ok(w.brier > 0.24 && w.brier < 0.26);
assert.ok(Math.abs(w.interval.c80 - 0.8) < 0.08, 'interval coverage is close to nominal on a stationary fixture');
assert.ok(s.latest && s.latest.q.length === 7);
assert.ok(JSON.stringify(s).length < 200000, 'summary stays small');
const sh = optimalShrink(J.resolved);
console.log('optimal shrink (ridge):', sh.lambda.toFixed(3), 'raw', sh.rawLambda.toFixed(3));

// the champion/challenger gate should (almost always) refuse to adopt on pure noise
const r = searchChallenger({ bars, champion: cfg.champion, seed: 1 });
console.log('search on noise:', r.decision, r.reason || '');
console.log('pipeline test OK');
