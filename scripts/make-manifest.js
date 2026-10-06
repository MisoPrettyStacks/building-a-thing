// Regenerates manifest.json (the list of files the page's "Download source files" button zips).
import fs from 'node:fs';
import path from 'node:path';
const root = path.resolve(import.meta.dirname, '..');
const out = [];
const walk = (d) => {
  for (const e of fs.readdirSync(path.join(root, d), { withFileTypes: true })) {
    const rel = path.posix.join(d, e.name);
    if (e.isDirectory()) { if (!['node_modules', '.git', 'data-branch'].includes(e.name)) walk(rel); }
    else out.push(rel);
  }
};
walk('.');
const files = out.map((f) => f.replace(/^\.\//, '')).filter((f) => f !== 'manifest.json' && !f.endsWith('.zip')).sort();
fs.writeFileSync(path.join(root, 'manifest.json'), JSON.stringify({ files }, null, 2) + '\n');
console.log(files.length + ' files in manifest');
