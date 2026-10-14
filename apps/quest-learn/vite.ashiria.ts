/**
 * Ashiria's development server routes and build guard.
 *
 * Development only (`npm run dev`):
 * - GET  /dev/packs/index.json and /dev/packs/<pack>/<file>: content packs from ../../content/packs, which may
 *   hold KSL Dictionary drafts. They are served to this Mac and the headset on the same Wi-Fi, never built
 *   into the app.
 * - POST /dev/studio: saves a Studio take (spec/STUDIO_FORMAT.md) to ../../training/data/studio/<session>/.
 *
 * Production builds only bundle packs from public/packs/, and refuse any that is not a release pack
 * (`ashiria pack --release`: Studio-recorded, approved signs only).
 */
import { existsSync, mkdirSync, readdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import type { IncomingMessage, ServerResponse } from 'node:http';
import { join, resolve } from 'node:path';
import type { Plugin } from 'vite';

const ROOT = resolve(__dirname, '../..');
const PACKS = join(ROOT, 'content/packs');
const STUDIO = join(ROOT, 'training/data/studio');
const SAFE = /^[a-z0-9][a-z0-9._-]*$/i;
const MAX_TAKE_BYTES = 20 * 1024 * 1024;

function send(res: ServerResponse, status: number, body: unknown, type = 'application/json'): void {
  res.statusCode = status;
  res.setHeader('Content-Type', type);
  res.setHeader('Cache-Control', 'no-store');
  res.end(typeof body === 'string' ? body : JSON.stringify(body));
}

function readBody(req: IncomingMessage): Promise<string> {
  return new Promise((ok, fail) => {
    const chunks: Buffer[] = [];
    let size = 0;
    req.on('data', (c: Buffer) => {
      size += c.length;
      if (size > MAX_TAKE_BYTES) { fail(new Error('take too large')); req.destroy(); return; }
      chunks.push(c);
    });
    req.on('end', () => ok(Buffer.concat(chunks).toString('utf8')));
    req.on('error', fail);
  });
}

/** Pack folders, newest first. */
function listPacks(dir: string): { id: string; manifest: unknown }[] {
  if (!existsSync(dir)) return [];
  return readdirSync(dir)
    .filter((d) => SAFE.test(d) && existsSync(join(dir, d, 'manifest.json')))
    .sort((a, b) => statSync(join(dir, b, 'manifest.json')).mtimeMs - statSync(join(dir, a, 'manifest.json')).mtimeMs)
    .map((id) => ({ id, manifest: JSON.parse(readFileSync(join(dir, id, 'manifest.json'), 'utf8')) }));
}

export function ashiriaDev(): Plugin {
  return {
    name: 'ashiria-dev',
    apply: 'serve',
    configureServer(server) {
      server.middlewares.use('/dev/packs', (req, res) => {
        const parts = (req.url ?? '/').split('?')[0].split('/').filter(Boolean);
        if (parts.length === 1 && parts[0] === 'index.json')
          return send(res, 200, listPacks(PACKS).map((p) => ({ id: p.id, distribution: (p.manifest as { distribution: string }).distribution })));
        if (parts.length !== 2 || !SAFE.test(parts[0]) || !['manifest.json', 'catalogue.json', 'references.json'].includes(parts[1]))
          return send(res, 404, { error: 'not found' });
        const file = join(PACKS, parts[0], parts[1]);
        if (!existsSync(file)) return send(res, 404, { error: 'not found' });
        return send(res, 200, readFileSync(file, 'utf8'));
      });

      server.middlewares.use('/dev/studio', async (req, res) => {
        if (req.method !== 'POST') return send(res, 405, { error: 'POST a take' });
        try {
          const text = await readBody(req);
          const take = JSON.parse(text) as { format?: string; session_id?: string; sign_id?: string };
          if (take.format !== 'ashiria.studio-take') return send(res, 400, { error: 'not a Studio take' });
          const session = String(take.session_id ?? ''), sign = String(take.sign_id ?? '');
          if (!SAFE.test(session) || !SAFE.test(sign)) return send(res, 400, { error: 'bad session or sign id' });
          const dir = join(STUDIO, session);
          mkdirSync(dir, { recursive: true });
          let n = 1;
          while (existsSync(join(dir, `${sign}__${n}.json`))) n++;
          const file = join(dir, `${sign}__${n}.json`);
          writeFileSync(file, text);
          return send(res, 200, { saved: file.slice(ROOT.length + 1), take: n });
        } catch (err) {
          return send(res, 400, { error: String(err) });
        }
      });
    },
  };
}

export function ashiriaReleaseGuard(): Plugin {
  return {
    name: 'ashiria-release-guard',
    apply: 'build',
    buildStart() {
      mkdirSync(join(__dirname, 'public/packs'), { recursive: true });
      const packs = listPacks(join(__dirname, 'public/packs'));
      for (const p of packs) {
        if ((p.manifest as { distribution?: string }).distribution !== 'release')
          this.error(`public/packs/${p.id} is not a release pack. Build it with \`ashiria pack --release\`; ` +
            'development packs (which may contain KSL Dictionary drafts) must never be bundled.');
      }
      writeFileSync(join(__dirname, 'public/packs/index.json'), JSON.stringify(packs.map((p) => ({ id: p.id, distribution: 'release' }))));
    },
  };
}
