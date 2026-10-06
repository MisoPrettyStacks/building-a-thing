// Real market data from free public exchange APIs (no keys). Works in browsers and Node >= 18.
// Primary: Coinbase Exchange XRP-USD. Cross-checks: Kraken and Bitstamp spot prices.
import { STEP } from './engine.js';

export const CB = 'https://api.exchange.coinbase.com';
export const PRODUCT = 'XRP-USD';

async function getJson(url, { timeout = 15000, retries = 3 } = {}) {
  let lastErr;
  for (let a = 0; a < retries; a++) {
    const ctl = new AbortController();
    const to = setTimeout(() => ctl.abort(), timeout);
    try {
      const res = await fetch(url, { signal: ctl.signal, headers: { Accept: 'application/json' } });
      clearTimeout(to);
      if (res.status === 429 || res.status >= 500) throw new Error('HTTP ' + res.status);
      if (!res.ok) throw new Error('HTTP ' + res.status + ' ' + url);
      return await res.json();
    } catch (e) {
      clearTimeout(to);
      lastErr = e;
      await new Promise((r) => setTimeout(r, 600 * (a + 1) ** 2));
    }
  }
  throw lastErr;
}

/** One request: up to ~300 candles. Returns ascending [{t,o,h,l,c,v}] with t = bucket start (unix s). */
export async function coinbaseCandles(startSec, endSec, granularity = STEP, product = PRODUCT) {
  const url = `${CB}/products/${product}/candles?granularity=${granularity}` +
    `&start=${new Date(startSec * 1000).toISOString()}&end=${new Date(endSec * 1000).toISOString()}`;
  const rows = await getJson(url);
  if (!Array.isArray(rows)) throw new Error('unexpected candle payload');
  return rows
    .map(([t, l, h, o, c, v]) => ({ t, o, h, l, c, v }))
    .sort((a, b) => a.t - b.t);
}

/** Closed candles only, paging backwards until `nBars` are collected. */
export async function fetchBars(nBars, { step = STEP, nowSec = Math.floor(Date.now() / 1000), product = PRODUCT } = {}) {
  const lastClosedStart = Math.floor(nowSec / step) * step - step;
  const per = 280;
  const map = new Map();
  let end = lastClosedStart;
  while (map.size < nBars) {
    const start = end - (per - 1) * step;
    const rows = await coinbaseCandles(start, end, step, product);
    for (const r of rows) if (r.t <= lastClosedStart) map.set(r.t, r);
    end = start - step;
    if (end < nowSec - 400 * 86400) break;
  }
  return [...map.values()].sort((a, b) => a.t - b.t).slice(-nBars);
}

/** Merge fresh closed bars into an existing ascending array (replaces bars with the same start time). */
export function mergeBars(existing, fresh, nowSec, step = STEP) {
  const m = new Map(existing.map((b) => [b.t, b]));
  for (const b of fresh) if (b.t + step <= nowSec) m.set(b.t, b);
  return [...m.values()].sort((a, b) => a.t - b.t);
}

export async function coinbaseTicker(product = PRODUCT) {
  const j = await getJson(`${CB}/products/${product}/ticker`, { retries: 2 });
  return { price: parseFloat(j.price), time: j.time, bid: parseFloat(j.bid), ask: parseFloat(j.ask) };
}

/** Independent spot prices from other venues, used for data-quality cross-checks. */
export async function referencePrices() {
  const out = {};
  await Promise.all([
    getJson('https://api.kraken.com/0/public/Ticker?pair=XRPUSD', { retries: 2 })
      .then((j) => { const k = Object.keys(j.result || {})[0]; out.kraken = parseFloat(j.result[k].c[0]); })
      .catch(() => {}),
    getJson('https://www.bitstamp.net/api/v2/ticker/xrpusd/', { retries: 2 })
      .then((j) => { out.bitstamp = parseFloat(j.last); })
      .catch(() => {}),
  ]);
  return out;
}
