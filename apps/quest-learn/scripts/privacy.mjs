// Builds public/privacy.html from docs/privacy.md (runs before every build), so the published policy and the
// repository copy never drift apart. Handles the small Markdown subset that document uses.
import { readFileSync, writeFileSync } from 'node:fs';

const md = readFileSync(new URL('../../../docs/privacy.md', import.meta.url), 'utf8');
const esc = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
const inline = (s) => esc(s)
  .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
  .replace(/\*(.+?)\*/g, '<em>$1</em>')
  .replace(/`(.+?)`/g, '<code>$1</code>')
  .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2">$1</a>');

const out = [];
const lines = md.split('\n');
for (let i = 0; i < lines.length; i++) {
  const line = lines[i];
  if (/^#{1,3} /.test(line)) { const n = line.match(/^#+/)[0].length; out.push(`<h${n}>${inline(line.slice(n + 1))}</h${n}>`); continue; }
  if (line.startsWith('|')) {
    const rows = [];
    while (i < lines.length && lines[i].startsWith('|')) rows.push(lines[i++]);
    i--;
    const cells = (r) => r.split('|').slice(1, -1).map((c) => c.trim());
    out.push('<table><thead><tr>' + cells(rows[0]).map((c) => `<th>${inline(c)}</th>`).join('') + '</tr></thead><tbody>' +
      rows.slice(2).map((r) => '<tr>' + cells(r).map((c) => `<td>${inline(c)}</td>`).join('') + '</tr>').join('') + '</tbody></table>');
    continue;
  }
  if (/^- /.test(line)) {
    const items = [];
    while (i < lines.length && /^- /.test(lines[i])) items.push(lines[i++].slice(2));
    i--;
    out.push('<ul>' + items.map((t) => `<li>${inline(t)}</li>`).join('') + '</ul>');
    continue;
  }
  if (!line.trim()) continue;
  const para = [line];
  while (i + 1 < lines.length && lines[i + 1].trim() && !/^(#|\||- )/.test(lines[i + 1])) para.push(lines[++i]);
  out.push(`<p>${inline(para.join(' '))}</p>`);
}

writeFileSync(new URL('../public/privacy.html', import.meta.url), `<!doctype html>
<html lang="en"><head><meta charset="utf-8" /><meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Ashiria Learn: privacy policy</title>
<style>
  body { margin: 0; background: #f6f7f8; color: #1d1d1f; font: 17px/1.55 -apple-system, system-ui, sans-serif; }
  main { max-width: 760px; margin: 0 auto; padding: 32px 20px 60px; }
  h1 { font-size: 30px; } h2 { margin-top: 32px; font-size: 21px; }
  table { border-collapse: collapse; width: 100%; font-size: 15px; } th, td { border: 1px solid #dbe0e4; padding: 8px 10px; text-align: left; vertical-align: top; }
  th { background: #eef2f4; } a { color: #0b7f99; }
  @media (prefers-color-scheme: dark) { body { background: #111315; color: #eef0f2; } th { background: #1f2327; } th, td { border-color: #2f3439; } a { color: #4fd1e8; } }
</style></head><body><main>
${out.join('\n')}
<p><a href="/">Back to Ashiria</a></p>
</main></body></html>
`);
console.log('public/privacy.html written from docs/privacy.md');
