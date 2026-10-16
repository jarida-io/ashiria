/** Loads the content pack: development packs from the dev server (this Mac only), release packs from the app bundle. */
import { parsePack, selfCheck, type ContentPack } from '@ashiria/engine';

const base = import.meta.env.DEV ? '/dev/packs' : `${import.meta.env.BASE_URL}packs`;

async function text(url: string): Promise<string> {
  const res = await fetch(url, { cache: 'no-store' });
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  return res.text();
}

/** A file the pack may not have (packs built before it existed). */
async function optional(url: string): Promise<string | null> {
  const res = await fetch(url, { cache: 'no-store' });
  return res.ok ? res.text() : null;
}

export async function loadPack(): Promise<ContentPack> {
  const index = JSON.parse(await text(`${base}/index.json`)) as { id: string; distribution: string }[];
  const choice = index.find((p) => p.id.startsWith('ksl-core')) ?? index[0];
  if (!choice) throw new Error(import.meta.env.DEV ? 'No content pack: run `make -C training pack`.' : 'No content pack in this build.');
  const file = (f: string) => `${base}/${choice.id}/${f}`;
  const [manifest, catalogue, references, coach] = await Promise.all([
    text(file('manifest.json')), text(file('catalogue.json')), text(file('references.json')), optional(file('coach.json'))]);
  const pack = await parsePack(manifest, catalogue, references, true, coach);
  const check = selfCheck(pack);
  if (!check.ok) throw new Error(`content pack self-check failed (${check.actual.toFixed(3)} vs ${check.expected})`);
  return pack;
}
