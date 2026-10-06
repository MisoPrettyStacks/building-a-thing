// UI smoke test: executes app.js against a stub DOM and fixture responses to catch runtime errors.
// Test-only. The fixtures here are never shipped as data and never reach a user.
import { forecastLatest, DEFAULT_CONFIG, QLEVELS } from '../lib/engine.js';
import { appendRecord, readLedger } from '../lib/io.js';
import { buildSummary } from '../lib/summary.js';
import { INITIAL_CONFIG, runAgent } from '../lib/agent.js';
import { joinLedger } from '../lib/summary.js';
import { mulberry32 } from '../lib/stats.js';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

// ---- fixture series & ledger
const rng = mulberry32(5);
const g = () => Math.sqrt(-2 * Math.log(1 - rng())) * Math.cos(2 * Math.PI * rng());
let lp = Math.log(0.5); const bars = [];
for (let i = 0; i < 3900; i++) { const o = Math.exp(lp); lp += 0.0012 * g(); const c = Math.exp(lp); bars.push({ t: 1.7e9 + i * 300, o, h: Math.max(o, c), l: Math.min(o, c), c, v: 1000 + 300 * rng() }); }
const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'ui-'));
const cfg = INITIAL_CONFIG();
for (let i = 3650; i < bars.length; i++) {
  const t = bars[i].t + 300, iso = new Date(t * 1000).toISOString();
  if (i - 3 >= 3650) { const f = bars[i - 3]; appendRecord(dir, { type: 'resolution', id: f.t, c1: bars[i].c, y: bars[i].c > f.c ? 1 : bars[i].c < f.c ? 0 : null, r: Math.log(bars[i].c / f.c) }, iso); }
  const { step } = forecastLatest(bars.slice(0, i + 1), cfg.champion);
  appendRecord(dir, { type: 'forecast', id: bars[i].t, bar_t: bars[i].t, t_issue: t, target_t: t + 900, p: step.p, p_raw: step.praw, m: step.m, q: step.q, nu: step.nu, c0: step.c0, cfg_version: 1, cfg_hash: 'abc', source: 'fixture' }, iso);
}
const records = readLedger(dir);
const nowSec = Math.floor(Date.now() / 1000);
const out = runAgent({ nowSec, resolved: joinLedger(records).resolved, bars, config: cfg });
const ag = out.events.map((e) => ({ ...e, ts: new Date().toISOString() }));
const summary = buildSummary({ records, config: out.config, agent: { events: ag, state: out.config.agent, history: [], previousVersion: null }, nowSec,
  extras: { health: { heartbeat: new Date().toISOString(), ledger: { ok: true, seq: records.length, head: 'ab'.repeat(32) }, bars_loaded: 100, bars_filled: 1, cross_check: { coinbase: 0.5, kraken: 0.5, bitstamp: 0.5, divergencePct: 0.01 } }, ledger_files: ['2026-10.jsonl'], model_state: { features: DEFAULT_CONFIG.features, w1: new Array(10).fill(0.01), weights: [0.25, 0.25, 0.25, 0.25], a: 1, b: 0, base: 0.5, m2: 1, m4: 6 } } });

// ---- stub DOM
const els = {};
const noopCtx = new Proxy(function () {}, { get: (_, k) => (k === 'measureText' ? () => ({ width: 10 }) : noopCtx), set: () => true, apply: () => noopCtx });
const mk = (id) => ({ id, style: {}, className: '', textContent: '', innerHTML: '', checked: true, clientWidth: 800, clientHeight: 300, width: 0, height: 0,
  classList: { toggle() {}, add() {}, remove() {} }, addEventListener() {}, getContext: () => noopCtx, appendChild() {}, click() {}, remove() {} });
globalThis.document = { getElementById: (id) => (els[id] ||= mk(id)), querySelectorAll: () => [], createElement: () => mk('x'), body: mk('body'), documentElement: {} };
globalThis.window = { devicePixelRatio: 1, addEventListener() {}, renderMathInElement: null };
globalThis.location = { hostname: 'example.com', pathname: '/', search: '' };
globalThis.getComputedStyle = () => ({ getPropertyValue: () => '#888888' });
globalThis.requestAnimationFrame = (fn) => fn();
globalThis.URL.createObjectURL = () => 'blob:x';
const errors = [];
process.on('unhandledRejection', (e) => errors.push(e));
const price = (t) => 0.5 + 0.02 * Math.sin(t / 40000) + 0.0015 * (mulberry32(t | 0)() - 0.5);
globalThis.fetch = async (url) => {
  url = String(url);
  const ok = (j) => ({ ok: true, status: 200, json: async () => j, text: async () => JSON.stringify(j), arrayBuffer: async () => new ArrayBuffer(0) });
  if (url.includes('summary.json')) return ok(summary);
  if (url.includes('config.json') && !url.includes('site-')) return ok(out.config);
  if (url.includes('site-config')) return { ok: false, status: 404, json: async () => ({}) };
  if (url.includes('/candles')) {
    const u = new URL(url); const gran = +u.searchParams.get('granularity');
    const s = Math.floor(Date.parse(u.searchParams.get('start')) / 1000), e = Math.floor(Date.parse(u.searchParams.get('end')) / 1000);
    const rows = [];
    for (let t = Math.ceil(s / gran) * gran; t <= e; t += gran) { const o = price(t), c = price(t + gran), hi = Math.max(o, c) + 0.0002, lo = Math.min(o, c) - 0.0002; rows.push([t, lo, hi, o, c, 1000 + (t % 7) * 50]); }
    return ok(rows.reverse());
  }
  return { ok: false, status: 404, json: async () => ({}) };
};
globalThis.WebSocket = undefined;
const origSetInterval = globalThis.setInterval; globalThis.setInterval = () => 0;

await import('../app.js');
await new Promise((r) => setTimeout(r, 6000));
const show = (id, n = 180) => console.log(`#${id}:`, String(els[id]?.innerHTML || els[id]?.textContent).replace(/\s+/g, ' ').slice(0, n));
show('hbText'); show('pUp'); show('fcI90'); show('scoreBody', 300); show('btStatus', 220); show('btBody', 300); show('agentLog', 300); show('intTable', 200); show('px');
if (errors.length) { console.error('UNHANDLED:', errors); process.exit(1); }
const must = [['scoreBody', 'Brier score'], ['btBody', 'Brier score'], ['agentLog', 'search'], ['paramTable', 'volLambda'], ['intTable', 'verified']];
for (const [id, s] of must) if (!String(els[id].innerHTML).includes(s)) { console.error('MISSING', s, 'in', id); process.exit(1); }
console.log('UI smoke test OK');
process.exit(0);
