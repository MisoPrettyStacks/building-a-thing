// Node-only helpers: tamper-evident append-only ledger (SHA-256 hash chain) and file access.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';

export function canonical(v) {
  if (v === null || typeof v !== 'object') return JSON.stringify(v);
  if (Array.isArray(v)) return '[' + v.map(canonical).join(',') + ']';
  return '{' + Object.keys(v).sort().filter((k) => v[k] !== undefined).map((k) => JSON.stringify(k) + ':' + canonical(v[k])).join(',') + '}';
}
export const sha256 = (s) => crypto.createHash('sha256').update(s).digest('hex');

const ledgerDir = (dir) => path.join(dir, 'ledger');
export function ledgerFiles(dir) {
  const d = ledgerDir(dir);
  if (!fs.existsSync(d)) return [];
  return fs.readdirSync(d).filter((f) => /^\d{4}-\d{2}\.jsonl$/.test(f)).sort();
}

export function readLedger(dir) {
  const out = [];
  for (const f of ledgerFiles(dir)) {
    for (const line of fs.readFileSync(path.join(ledgerDir(dir), f), 'utf8').split('\n')) {
      if (line.trim()) out.push(JSON.parse(line));
    }
  }
  return out;
}

function head(dir) {
  const p = path.join(dir, 'head.json');
  return fs.existsSync(p) ? JSON.parse(fs.readFileSync(p, 'utf8')) : { seq: 0, hash: '0'.repeat(64) };
}

/** Append one event. hash = SHA256(prev_hash + canonical(record without hash)). */
export function appendRecord(dir, payload, tsIso = new Date().toISOString()) {
  fs.mkdirSync(ledgerDir(dir), { recursive: true });
  const h = head(dir);
  const rec = { seq: h.seq + 1, ts: tsIso, ...payload, prev: h.hash };
  rec.hash = sha256(h.hash + canonical(rec));
  const month = tsIso.slice(0, 7);
  fs.appendFileSync(path.join(ledgerDir(dir), `${month}.jsonl`), JSON.stringify(rec) + '\n');
  fs.writeFileSync(path.join(dir, 'head.json'), JSON.stringify({ seq: rec.seq, hash: rec.hash }));
  return rec;
}

export function verifyChain(dir) {
  let prev = '0'.repeat(64), seq = 0;
  for (const rec of readLedger(dir)) {
    const { hash, ...rest } = rec;
    if (rec.prev !== prev || rec.seq !== seq + 1 || sha256(prev + canonical(rest)) !== hash) {
      return { ok: false, brokenAt: rec.seq };
    }
    prev = hash; seq = rec.seq;
  }
  return { ok: true, seq, head: prev };
}

export function readJson(file, fallback) {
  try { return JSON.parse(fs.readFileSync(file, 'utf8')); } catch { return fallback; }
}
export function writeJson(file, obj) {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  const tmp = file + '.tmp';
  fs.writeFileSync(tmp, JSON.stringify(obj));
  fs.renameSync(tmp, file);
}
