/**
 * Ashiria's development server routes and build guard.
 *
 * Development only (`npm run dev`):
 * - GET  /dev/packs/index.json and /dev/packs/<pack>/<file>: content packs from ../../content/packs, which may
 *   hold KSL Dictionary drafts. They are served to this Mac and the headset on the same Wi-Fi, never built
 *   into the app.
 * - POST /dev/studio: saves a Studio take (spec/STUDIO_FORMAT.md) to ../../training/data/studio/<session>/.
 * - /dev/study: the study sheet (dev/study.html): each sign's local dictionary video next to its reference hands,
 *   with Approve / Reject buttons that run `ashiria approve`. /dev/study/browse: every sign in the dictionary's data
 *   sheet, with MediaPipe's landmarks drawn over its video, the reference Ashiria would cut from it, audit flags and
 *   the recogniser's results (signs.json of the newest training/runs/embed- run), and an avatar (dev/avatar.js)
 *   that signs each clip from its landmarks. The videos never leave this Mac.
 *
 * Production builds only bundle packs from public/packs/, and refuse any that is not a release pack
 * (`ashiria pack --release`: Studio-recorded, approved signs only).
 */
import { execFile } from 'node:child_process';
import { createReadStream, existsSync, mkdirSync, readdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import type { IncomingMessage, ServerResponse } from 'node:http';
import { join, resolve } from 'node:path';
import type { Plugin } from 'vite';

const ROOT = resolve(__dirname, '../..');
const PACKS = join(ROOT, 'content/packs');
const STUDIO = join(ROOT, 'training/data/studio');
const TRAINING = join(ROOT, 'training');
const ASHIRIA = join(TRAINING, '.venv/bin/ashiria');
const VIDEO_ROOT = join(TRAINING, 'data');
const RUNS = join(TRAINING, 'runs');
const REVIEW = ['approved', 'rejected', 'draft'];
// The sign browser's index is rebuilt when these change (re-import or re-audit of the dictionary).
const INDEX_INPUTS = ['dictionary_labels.csv', 'dictionary_audit.csv'].map((f) => join(TRAINING, 'data/dictionary/manifests', f));

function ashiria(args: string[]): Promise<string> {
  return new Promise((ok, fail) => execFile(ASHIRIA, args, { cwd: TRAINING, maxBuffer: 64 * 1024 * 1024 },
    (err, stdout, stderr) => (err ? fail(new Error(stderr || String(err))) : ok(stdout))));
}

/** Stream a local video with HTTP range support (browsers need it to play <video>). */
function sendVideo(req: IncomingMessage, res: ServerResponse, file: string): void {
  const size = statSync(file).size;
  const range = /bytes=(\d*)-(\d*)/.exec(req.headers.range ?? '');
  res.setHeader('Content-Type', 'video/mp4');
  res.setHeader('Accept-Ranges', 'bytes');
  if (range) {
    const start = range[1] ? Number(range[1]) : 0;
    const end = range[2] ? Math.min(Number(range[2]), size - 1) : size - 1;
    res.statusCode = 206;
    res.setHeader('Content-Range', `bytes ${start}-${end}/${size}`);
    res.setHeader('Content-Length', end - start + 1);
    createReadStream(file, { start, end }).pipe(res);
  } else {
    res.setHeader('Content-Length', size);
    createReadStream(file).pipe(res);
  }
}
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

      let videos = new Map<string, string>();
      let clips = new Map<string, string>();
      let indexText = '';
      let indexStamp = '';
      const landmarkCache = new Map<string, string>();
      const referenceCache = new Map<string, string>();
      const poseCache = new Map<string, string>();
      server.middlewares.use('/dev/study', async (req, res) => {
        const url = (req.url ?? '/').split('?')[0];
        try {
          if (url === '/' || url === '') return send(res, 200, readFileSync(join(__dirname, 'dev/study.html'), 'utf8'), 'text/html');
          if (url === '/browse' || url === '/browse/') return send(res, 200, readFileSync(join(__dirname, 'dev/browse.html'), 'utf8'), 'text/html');
          if (url === '/index.json') {
            const stamp = INDEX_INPUTS.map((f) => (existsSync(f) ? statSync(f).mtimeMs : 0)).join();
            if (stamp !== indexStamp) [indexText, indexStamp] = ['', stamp];
            if (!indexText) {
              indexText = await ashiria(['study-index']);
              const entries = JSON.parse(indexText) as { clip_id: string | null; video: string | null }[];
              clips = new Map(entries.filter((c) => c.clip_id && c.video).map((c) => [c.clip_id as string, c.video as string]));
            }
            return send(res, 200, indexText);
          }
          if (url === '/recogniser.json') {
            // Per-sign results of the newest `ashiria embed` run (`ashiria embed-signs`), if any.
            const runs = existsSync(RUNS) ? readdirSync(RUNS).filter((d) => d.startsWith('embed-') && existsSync(join(RUNS, d, 'signs.json'))).sort() : [];
            if (!runs.length) return send(res, 404, { error: 'no recogniser results yet: run `ashiria embed-signs`' });
            return send(res, 200, readFileSync(join(RUNS, runs[runs.length - 1], 'signs.json'), 'utf8'));
          }
          const lm = /^\/landmarks\/([a-f0-9]+)$/.exec(url);
          if (lm) {
            if (!landmarkCache.has(lm[1])) {
              if (landmarkCache.size > 200) landmarkCache.clear();
              landmarkCache.set(lm[1], await ashiria(['study-landmarks', lm[1]]));
            }
            return send(res, 200, landmarkCache.get(lm[1]) as string);
          }
          if (url === '/avatar.js') return send(res, 200, readFileSync(join(__dirname, 'dev/avatar.js'), 'utf8'), 'text/javascript');
          const pose = /^\/pose\/([a-f0-9]+)$/.exec(url);
          if (pose) {
            if (!poseCache.has(pose[1])) {
              if (poseCache.size > 200) poseCache.clear();
              poseCache.set(pose[1], await ashiria(['study-pose', pose[1]]));
            }
            const text = poseCache.get(pose[1]) as string;
            return text.trim() === 'null' ? send(res, 404, { error: 'no body pose' }) : send(res, 200, text);
          }
          const ref = /^\/reference\/([a-f0-9]+)$/.exec(url);
          if (ref) {
            if (!referenceCache.has(ref[1])) {
              if (referenceCache.size > 200) referenceCache.clear();
              referenceCache.set(ref[1], await ashiria(['study-reference', ref[1]]));
            }
            const text = referenceCache.get(ref[1]) as string;
            return text.trim() === 'null' ? send(res, 404, { error: 'no reference' }) : send(res, 200, text);
          }
          const clip = /^\/clip\/([a-f0-9]+)$/.exec(url);
          if (clip) {
            const file = clips.get(clip[1]);
            if (!file || !resolve(file).startsWith(VIDEO_ROOT + '/') || !existsSync(file)) return send(res, 404, { error: 'no video' });
            return sendVideo(req, res, file);
          }
          if (url === '/data.json') {
            const text = await ashiria(['study-data']);
            const data = JSON.parse(text) as { signs: { id: string; video: string | null }[] };
            videos = new Map(data.signs.filter((s) => s.video).map((s) => [s.id, s.video as string]));
            return send(res, 200, text);
          }
          const video = /^\/video\/([a-z0-9-]+)$/.exec(url);
          if (video) {
            const file = videos.get(video[1]);
            if (!file || !resolve(file).startsWith(VIDEO_ROOT + '/') || !existsSync(file)) return send(res, 404, { error: 'no video' });
            return sendVideo(req, res, file);
          }
          if (url === '/review' && req.method === 'POST') {
            const { id, status } = JSON.parse(await readBody(req)) as { id?: string; status?: string };
            if (!id || !SAFE.test(id) || !status || !REVIEW.includes(status)) return send(res, 400, { error: 'bad review' });
            return send(res, 200, { result: (await ashiria(['approve', id, '--status', status])).trim() });
          }
          return send(res, 404, { error: 'not found' });
        } catch (err) {
          return send(res, 500, { error: String(err) });
        }
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
