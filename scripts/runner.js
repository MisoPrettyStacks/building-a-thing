// Continuous runner. Intended for GitHub Actions (free for public repos) but runs anywhere with Node >= 18.
//   node scripts/runner.js --data ./data-branch --minutes 305 --push
// Each closed 5-minute bar: resolve due forecasts -> issue the next forecast -> run the agent -> publish scoreboard.
import { execSync } from 'node:child_process';
import path from 'node:path';
import { fetchBars, mergeBars, coinbaseCandles, referencePrices } from '../lib/data.js';
import { gridBars, forecastLatest, STEP, QLEVELS } from '../lib/engine.js';
import { appendRecord, readLedger, readJson, writeJson, verifyChain, canonical, sha256, ledgerFiles } from '../lib/io.js';
import { buildSummary, joinLedger } from '../lib/summary.js';
import { runAgent, INITIAL_CONFIG } from '../lib/agent.js';

const arg = (k, d) => { const i = process.argv.indexOf('--' + k); return i < 0 ? d : (process.argv[i + 1]?.startsWith('--') || i + 1 >= process.argv.length ? true : process.argv[i + 1]); };
const DIR = path.resolve(arg('data', './data-branch'));
const MINUTES = Number(arg('minutes', 0));
const PUSH = process.argv.includes('--push');
const ALLOW_LATE = process.argv.includes('--allow-late');
const HIST_BARS = Number(arg('history-bars', 6048)); // 21 days
const MAX_LATE_SEC = 150;
const startedAt = Date.now();
const now = () => Math.floor(Date.now() / 1000);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const log = (...a) => console.log(new Date().toISOString(), ...a);

const sh = (c) => execSync(c, { cwd: DIR, stdio: ['ignore', 'pipe', 'pipe'] }).toString();
let lastPush = 0;
function publish(force = false) {
  if (!PUSH) return;
  if (!force && Date.now() - lastPush < 4.5 * 60 * 1000) return;
  try {
    sh('git add -A');
    if (!sh('git status --porcelain').trim()) return;
    let has = true;
    try { sh('git rev-parse HEAD'); } catch { has = false; }
    if (has) sh(`git commit --amend -q -m "data ${new Date().toISOString()}"`);
    else sh(`git commit -q -m "data ${new Date().toISOString()}"`);
    sh('git push --force -q origin HEAD:data');
    lastPush = Date.now();
  } catch (e) { log('publish failed:', String(e.stderr || e).slice(0, 300)); }
}

let config = readJson(path.join(DIR, 'config.json'), null) || INITIAL_CONFIG();
let records = readLedger(DIR);
const cfgHash = () => sha256(canonical(config.champion)).slice(0, 16);
const saveConfig = () => writeJson(path.join(DIR, 'config.json'), config);
const chain = verifyChain(DIR);
if (!chain.ok) { log('LEDGER CHAIN BROKEN at seq', chain.brokenAt, '- refusing to continue'); process.exit(2); }

let raw = [];
let bars = [];
const stats = { cycles: 0, errors: 0, lastRef: null, lastError: null };

function append(payload) {
  const rec = appendRecord(DIR, payload);
  records.push(rec);
  return rec;
}

async function cycle() {
  const t = now();
  const lastClosedStart = Math.floor(t / STEP) * STEP - STEP;
  const fresh = await coinbaseCandles(t - 30 * STEP, t);
  raw = mergeBars(raw, fresh, t).slice(-(HIST_BARS + 600));
  bars = gridBars(raw, STEP, lastClosedStart);
  const last = bars[bars.length - 1];
  if (!last || last.t !== lastClosedStart) throw new Error('latest closed bar missing');

  const J = joinLedger(records);
  const h = config.champion.h;

  // 1) resolve forecasts whose target bar has closed
  for (const f of J.pending) {
    const targetBarStart = f.target_t - STEP;
    if (targetBarStart > lastClosedStart) continue;
    const b = bars.find((x) => x.t === targetBarStart) || null;
    if (!b) {
      if (t > f.target_t + 3600) append({ type: 'void', id: f.id, reason: 'target bar unavailable after 1h' });
      continue;
    }
    const y = b.c > f.c0 ? 1 : b.c < f.c0 ? 0 : null;
    append({ type: 'resolution', id: f.id, c1: b.c, y, r: Math.log(b.c / f.c0), filled: !!b.filled, source: 'coinbase:XRP-USD' });
    log(`resolved ${f.id}: p=${f.p} c0=${f.c0} c1=${b.c} y=${y}`);
  }

  // 2) issue the forecast for the bar that just closed
  const exists = records.some((r) => r.type === 'forecast' && r.id === lastClosedStart) ||
    records.some((r) => r.type === 'gap' && r.bar_t === lastClosedStart);
  if (!exists) {
    const lag = t - (lastClosedStart + STEP);
    if (lag > MAX_LATE_SEC && !ALLOW_LATE) {
      append({ type: 'gap', bar_t: lastClosedStart, reason: `runner late by ${lag}s (> ${MAX_LATE_SEC}s)` });
      log('skipped bar', lastClosedStart, 'late by', lag);
    } else {
      const { step, state } = forecastLatest(bars, config.champion);
      const rec = append({
        type: 'forecast', id: lastClosedStart, bar_t: lastClosedStart, t_issue: lastClosedStart + STEP,
        target_t: lastClosedStart + STEP + h * STEP, issue_lag_sec: lag,
        p: +step.p.toFixed(6), p_raw: +step.praw.toFixed(6), m: step.m.map((x) => +x.toFixed(6)),
        q: step.q.map((x) => +x.toFixed(7)), ladder: step.ladder.map((x) => +x.toFixed(5)), q_levels: QLEVELS, nu: step.nu, c0: step.c0,
        cfg_version: config.champion.version, cfg_hash: cfgHash(),
        input_digest: sha256(canonical(bars.slice(-48).map((b) => [b.t, b.c, b.v]))).slice(0, 16),
        source: 'coinbase:XRP-USD', weights: state.weights.map((x) => +x.toFixed(4)),
      });
      log(`forecast ${rec.id}: P(up)=${rec.p} c0=${rec.c0} v${rec.cfg_version}`);
      stats.lastState = state;
    }
  }

  // 3) agent review
  const resolved = joinLedger(records).resolved;
  const before = canonical(config);
  const out = runAgent({ nowSec: t, resolved, bars, config });
  config = out.config;
  for (const ev of out.events) { append({ type: 'agent', ...ev }); log('agent:', ev.type, ev.decision || ev.action || ''); }
  if (canonical(config) !== before) saveConfig();

  // 4) cross-venue sanity check
  try {
    const ref = await referencePrices();
    const vals = Object.values(ref).filter(Number.isFinite);
    if (vals.length) {
      const med = vals.sort((a, b) => a - b)[Math.floor(vals.length / 2)];
      stats.lastRef = { ...ref, coinbase: last.c, divergencePct: (Math.abs(last.c / med - 1) * 100) };
    }
  } catch { /* optional */ }
  stats.cycles++;
  stats.lastError = null;
}

function writeSummary() {
  const agentEvents = records.filter((r) => r.type === 'agent').slice(-40).reverse().map((r) => {
    const { prev, hash, ...rest } = r; return rest;
  });
  const chk = verifyChain(DIR);
  const J = joinLedger(records);
  const summary = buildSummary({
    records, config, agent: { events: agentEvents, state: config.agent, history: config.history.slice(-30).reverse(), previousVersion: config.previous?.version ?? null },
    extras: {
      health: {
        runner_started: new Date(startedAt).toISOString(), heartbeat: new Date().toISOString(), cycles: stats.cycles,
        last_error: stats.lastError, ledger: chk, bars_loaded: bars.length, bars_filled: bars.filter((b) => b.filled).length,
        cross_check: stats.lastRef, source: 'Coinbase Exchange XRP-USD 5-minute candles',
        commit: process.env.GITHUB_SHA || null, tie_count: J.ties,
      },
      ledger_files: ledgerFiles(DIR),
      model_state: stats.lastState || null,
    },
  });
  writeJson(path.join(DIR, 'summary.json'), summary);
}

async function main() {
  log(`runner start: dir=${DIR} minutes=${MINUTES} push=${PUSH} ledger_seq=${chain.seq || 0}`);
  raw = await fetchBars(HIST_BARS);
  log(`history loaded: ${raw.length} closed 5-min bars`);
  const deadline = MINUTES > 0 ? startedAt + MINUTES * 60000 : 0;
  for (;;) {
    try {
      await cycle();
      writeSummary();
      publish();
    } catch (e) {
      stats.errors++; stats.lastError = String(e.message || e).slice(0, 200);
      log('cycle error:', stats.lastError);
      try { writeSummary(); } catch { /* ignore */ }
    }
    if (!deadline) break;
    // sleep until 6s after the next 5-minute boundary; retry quickly after an error
    const t = Date.now() / 1000;
    let waitS = stats.lastError ? 15 : (STEP - (t % STEP)) + 6;
    if (Date.now() + waitS * 1000 > deadline) break;
    await sleep(waitS * 1000);
  }
  writeSummary();
  publish(true);
  log('runner finished');
}
main().catch((e) => { console.error(e); process.exit(1); });
