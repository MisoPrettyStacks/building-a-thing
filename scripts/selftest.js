// Unit tests for the math and the no-leakage guarantee.
// The deterministic series built here exist ONLY to test code properties. They are never displayed,
// scored, shipped as data, or used by the live system, which only ever sees real exchange candles.
import assert from 'node:assert/strict';
import { normCdf, tCdf, tQuantile, betaInc, lgamma, brier, logloss, dmTest, wilson, overlapDeff,
  binaryScores, calibrationFit, mulberry32, sigmoid, logit } from '../lib/stats.js';
import { walkForward, forecastLatest, DEFAULT_CONFIG, gridBars } from '../lib/engine.js';
import { appendRecord, verifyChain } from '../lib/io.js';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

let passed = 0;
const near = (a, b, tol, msg) => { assert.ok(Math.abs(a - b) <= tol, `${msg}: ${a} vs ${b}`); passed++; };
const ok = (c, msg) => { assert.ok(c, msg); passed++; };

// --- special functions against published values
near(normCdf(1.959964), 0.975, 1e-6, 'normCdf(1.96)');
near(normCdf(0), 0.5, 1e-15, 'normCdf(0)');
near(lgamma(5), Math.log(24), 1e-12, 'lgamma(5)');
near(betaInc(0.5, 2, 3), 0.6875, 1e-12, 'I_0.5(2,3)');
near(tCdf(2.0, 5), 0.9490303, 1e-6, 'tCdf(2,5)');
near(tQuantile(0.975, 10), 2.228139, 1e-5, 't_0.975,10');
near(tQuantile(0.95, 3), 2.353363, 1e-5, 't_0.95,3');
near(tQuantile(0.025, 10), -2.228139, 1e-5, 't symmetry');
near(tQuantile(0.9, 30), 1.310415, 1e-5, 't_0.9,30');
near(tCdf(1.5, 1e6), normCdf(1.5), 1e-5, 't -> normal');

// --- scores
near(brier(0.7, 1), 0.09, 1e-12, 'brier');
near(logloss(0.5, 1), Math.LN2, 1e-12, 'logloss');
near(overlapDeff(3), 1 + 2 * ((2 / Math.PI) * Math.asin(2 / 3) + (2 / Math.PI) * Math.asin(1 / 3)), 1e-12, 'Deff');
{
  const [lo, hi] = wilson(50, 100);
  near(lo, 0.4038, 1e-3, 'wilson lo'); near(hi, 0.5962, 1e-3, 'wilson hi');
}
{ // DM: identical losses -> stat 0
  const a = Array.from({ length: 100 }, (_, i) => (i % 7) / 7);
  const r = dmTest(a, a);
  near(r.dbar, 0, 1e-15, 'dm zero'); ok(!(r.pALess < 0.01), 'dm no signal');
  // A uniformly lower loss -> A better
  const b = a.map((x) => x + 0.05 + 0.01 * Math.sin(x * 100));
  ok(dmTest(a, b).pALess < 0.001, 'dm detects better forecaster');
}
{ // calibration recovers a known relationship
  const rng = mulberry32(7);
  const ps = [], ys = [];
  for (let i = 0; i < 20000; i++) { const p = 0.2 + 0.6 * rng(); ps.push(p); ys.push(rng() < p ? 1 : 0); }
  const c = calibrationFit(ps, ys);
  near(c.beta, 1, 0.15, 'calibration slope of calibrated forecasts');
  near(c.alpha, 0, 0.08, 'calibration intercept');
  const s = binaryScores(ps, ys);
  ok(s.ece < 0.02, 'ECE small for calibrated forecasts');
}

// --- deterministic test series (LCG, Box-Muller): NOT data, only a fixture
function fixture(n, { phi = 0, seed = 11, sigma = 0.0012 } = {}) {
  const rng = mulberry32(seed);
  const gauss = () => Math.sqrt(-2 * Math.log(1 - rng())) * Math.cos(2 * Math.PI * rng());
  let lp = Math.log(0.5), prev = 0;
  const bars = [];
  for (let i = 0; i < n; i++) {
    const r = phi * prev + sigma * gauss();
    prev = r;
    const o = Math.exp(lp); lp += r; const c = Math.exp(lp);
    bars.push({ t: 1.7e9 + i * 300, o, h: Math.max(o, c), l: Math.min(o, c), c, v: 1000 + 200 * rng() });
  }
  return bars;
}

// --- no look-ahead: changing the future must not change any earlier forecast
{
  const a = fixture(1500);
  const b = a.map((x) => ({ ...x }));
  const cut = 1000;
  const rng = mulberry32(99);
  for (let i = cut + 1; i < b.length; i++) { const c = b[i].c * (1 + 0.05 * (rng() - 0.5)); b[i] = { ...b[i], c, o: c, h: c, l: c, v: 5 + 5000 * rng() }; }
  const ra = walkForward(a, DEFAULT_CONFIG, { quantFrom: 0 }).steps;
  const rb = walkForward(b, DEFAULT_CONFIG, { quantFrom: 0 }).steps;
  let compared = 0;
  for (let k = 0; k < ra.length; k++) {
    if (ra[k].i > cut) break;
    assert.equal(ra[k].p, rb[k].p, `forecast ${ra[k].i} changed when the future changed`);
    assert.deepEqual(ra[k].q, rb[k].q, `quantiles ${ra[k].i} changed`);
    compared++;
  }
  ok(compared > 600, 'leakage test compared many forecasts');
}

// --- determinism, and live path == backtest path
{
  const a = fixture(1400);
  const r1 = walkForward(a, DEFAULT_CONFIG, { quantFrom: 1399 });
  const r2 = walkForward(a, DEFAULT_CONFIG, { quantFrom: 1399 });
  assert.deepEqual(r1.steps.at(-1), r2.steps.at(-1)); passed++;
  const f = forecastLatest(a, DEFAULT_CONFIG);
  assert.equal(f.step.p, r1.steps.at(-1).p); passed++;
  // forecast at index i computed on a truncated series equals the one computed inside the full replay
  const full = walkForward(a, DEFAULT_CONFIG).steps;
  const trunc = walkForward(a.slice(0, 1200), DEFAULT_CONFIG).steps;
  assert.equal(trunc.at(-1).p, full.find((s) => s.i === 1199).p); passed++;
}

// --- does no harm on a driftless random walk (should score ~0.25, not worse)
{
  const a = fixture(6000, { seed: 3 });
  const s = walkForward(a, DEFAULT_CONFIG).steps.filter((x) => x.y !== null && x.i > 1500);
  const bs = s.reduce((t, x) => t + brier(x.p, x.y), 0) / s.length;
  ok(bs < 0.2515 && bs > 0.245, `random-walk Brier near 0.25 (got ${bs.toFixed(5)})`);
  const mx = Math.max(...s.map((x) => Math.abs(x.p - 0.5)));
  ok(mx < 0.2, `no wild overconfidence on noise (max |p-0.5| = ${mx.toFixed(3)})`);
}

// --- can learn a real signal when one exists (persistent returns)
{
  const phi = 0.35, sigma = 0.0012;
  const a = fixture(6000, { phi, seed: 5, sigma });
  const s = walkForward(a, DEFAULT_CONFIG).steps.filter((x) => x.y !== null && x.i > 1500);
  const bs = s.reduce((t, x) => t + brier(x.p, x.y), 0) / s.length;
  // exact oracle for AR(1) returns: P(sum of next 3 returns > 0 | r_t)
  const sd = sigma * Math.sqrt((1 + phi + phi * phi) ** 2 + (1 + phi) ** 2 + 1);
  let ob = 0;
  for (const x of s) {
    const rt = Math.log(a[x.i].c / a[x.i - 1].c);
    const po = normCdf((rt * (phi + phi * phi + phi ** 3)) / sd);
    ob += brier(po, x.y);
  }
  ob /= s.length;
  ok(ob < 0.2475, `oracle sanity (${ob.toFixed(4)})`);
  ok(bs < 0.2475 && bs - ob < 0.004, `learns planted autocorrelation: engine ${bs.toFixed(4)} vs oracle ${ob.toFixed(4)}`);
}

// --- gap filling
{
  const raw = [{ t: 0, o: 1, h: 1, l: 1, c: 1, v: 5 }, { t: 900, o: 2, h: 2, l: 2, c: 2, v: 5 }];
  const g = gridBars(raw);
  assert.equal(g.length, 4); assert.equal(g[1].c, 1); assert.equal(g[3].c, 2); passed += 3;
}

// --- tamper-evident ledger
{
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'ledger-'));
  for (let i = 0; i < 5; i++) appendRecord(dir, { type: 'forecast', id: i, p: 0.5 + i / 100 }, '2026-10-05T00:00:00Z');
  ok(verifyChain(dir).ok, 'chain verifies');
  const f = path.join(dir, 'ledger', '2026-10.jsonl');
  const lines = fs.readFileSync(f, 'utf8').trim().split('\n');
  lines[2] = lines[2].replace('"p":0.52', '"p":0.99');
  fs.writeFileSync(f, lines.join('\n') + '\n');
  ok(!verifyChain(dir).ok, 'tampering is detected');
}

console.log(`selftest: ${passed} checks passed`);
