# XRP 15-Minute Forecast

A calibrated, publicly scored probability that XRP/USD is higher 15 minutes from now, issued every 5 minutes,
with a live chart, full methodology/maths, accuracy statistics (Brier score, skill, log loss, reliability, ECE,
Diebold-Mariano tests, interval coverage) and an autonomous accuracy agent. Runs free, 24/7, on GitHub Actions + Pages.
No API keys, no paid services, no mock data: only real Coinbase candles (cross-checked against Kraken and Bitstamp).

## Deploy (about 5 minutes, free)
1. Create a **public** GitHub repository and push this folder to its `main` branch.
2. Settings -> Pages -> Build and deployment -> Source: *Deploy from a branch* -> `main` / root.
3. Settings -> Actions -> General -> Workflow permissions: **Read and write**.
4. Actions tab -> `forecast-runner` -> *Run workflow*. It then re-schedules itself every 5 hours and loops continuously.
5. Open `https://<user>.github.io/<repo>/`. The page finds the data branch automatically. For any other host, put
   `"repo": "<user>/<repo>"` in `site-config.json`.

Layout: code lives on `main`; the runner publishes `summary.json`, `config.json` and the hash-chained `ledger/*.jsonl`
to an orphan `data` branch (kept as a single amended commit so history never bloats).

## Files
- `lib/engine.js` features, models, pooling, calibration, predictive distribution (`walkForward`, the single code path)
- `lib/stats.js` special functions, scoring rules, DM test, bootstrap, calibration
- `lib/agent.js` accuracy agent (shrinkage guard, champion/challenger, rollback)
- `lib/data.js` Coinbase / Kraken / Bitstamp public API access
- `lib/io.js`, `lib/summary.js` hash-chained ledger and scoreboard
- `scripts/runner.js` the loop; `scripts/selftest*.js`, `scripts/smoke-ui.js` tests (deterministic fixtures, test-only)
- `index.html`, `app.js`, `style.css` the page

## Verify
`npm test` (Node >= 18). Verify the ledger chain with `verifyChain()` from `lib/io.js` against a clone of the `data` branch.

## Honest expectations
Short-horizon crypto direction is close to a martingale. Expect Brier skill of at most a few percent over 0.25 and a hit rate
near 50%. The system is built to stay calibrated, to say so when there is no significant skill, and to shrink its
confidence automatically. GitHub's scheduler is best effort: skipped bars are recorded as gaps, never invented.
