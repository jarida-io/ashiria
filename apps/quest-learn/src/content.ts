/** Loads the content pack: development packs from the dev server (this Mac only), release packs from the app bundle. */
import { parsePack, selfCheck, type ContentPack } from '@ashiria/engine';

const base = import.meta.env.DEV ? '/dev/packs' : `${import.meta.env.BASE_URL}packs`;

async function text(url: string): Promise<string> {
  const res = await fetch(url, { cache: 'no-store' });
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  return res.text();
}

export async function loadPack(): Promise<ContentPack> {
  const index = JSON.parse(await text(`${base}/index.json`)) as { id: string; distribution: string }[];
  const choice = index.find((p) => p.id.startsWith('ksl-core')) ?? index[0];
  if (!choice) throw new Error(import.meta.env.DEV ? 'No content pack: run `make -C training pack`.' : 'No content pack in this build.');
  const [manifest, catalogue, references] = await Promise.all(
    ['manifest.json', 'catalogue.json', 'references.json'].map((f) => text(`${base}/${choice.id}/${f}`)));
  const pack = await parsePack(manifest, catalogue, references);
  const check = selfCheck(pack);
  if (!check.ok) throw new Error(`content pack self-check failed (${check.actual.toFixed(3)} vs ${check.expected})`);
  return pack;
}
