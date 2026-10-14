/** Content pack loader (spec/CONTENT_PACK.md; port of ContentPack.cs). */
import { SPEC_VERSION } from './features.ts';
import { paramsFromJson, RUNTIME_VERSION, type EngineParams } from './params.ts';
import { segmentFromFlat } from './segment.ts';
import { bindTake, type SignRef, type Tolerances } from './signs.ts';

export const FORMAT = 'ashiria.content-pack';
export const FORMAT_VERSION = 1;

export interface CatalogueSign {
  id: string; gloss: string; aliases: string[]; unit: string; order: number;
  tier: 'scored' | 'lenient' | 'watch'; tier_source: string; tier_reasons: string[];
  recognise: boolean; hint: string; facial_grammar: boolean;
  tags: { hands: string; movement: string; near_face: boolean; hands_close: boolean };
  lookalikes: string[]; source: { kind: string; takes: number; signers: string[] };
  review: { status: string; by?: string | null; date?: string | null };
}

export interface Unit { id: string; title: string; order: number }

export interface ContentPack {
  packId: string; version: string; distribution: 'internal' | 'release'; contentHash: string;
  params: EngineParams; units: Unit[]; catalogue: CatalogueSign[]; signs: Map<string, SignRef>;
  selfcheck: { sign_id: string; take: number; descriptor_sum: number; tolerance: number };
}

async function sha256(text: string): Promise<string> {
  const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(text));
  return Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, '0')).join('');
}

/** Parse a pack from the text of its three files. Throws on a format, version or hash mismatch. */
export async function parsePack(manifestText: string, catalogueText: string, referencesText: string, checkHashes = true): Promise<ContentPack> {
  const m: any = JSON.parse(manifestText);
  if (m.format !== FORMAT) throw new Error('not an Ashiria content pack');
  if (m.format_version !== FORMAT_VERSION) throw new Error(`content pack format ${m.format_version}; this app reads ${FORMAT_VERSION}`);
  if (m.spec_version !== SPEC_VERSION) throw new Error(`pack spec_version ${m.spec_version}; this engine is ${SPEC_VERSION}`);
  if (m.runtime_version !== RUNTIME_VERSION) throw new Error(`pack runtime_version ${m.runtime_version}; this engine is ${RUNTIME_VERSION}`);
  if (checkHashes) {
    for (const [file, text] of [['catalogue', catalogueText], ['references', referencesText]] as const)
      if ((await sha256(text)) !== m.files[file].sha256) throw new Error(`${m.files[file].path} does not match the manifest (damaged or edited)`);
  }
  const params = paramsFromJson(m.engine);
  const cat = JSON.parse(catalogueText) as { units: Unit[]; signs: CatalogueSign[] };
  const refs = JSON.parse(referencesText) as { signs: Record<string, { takes: any[]; shadow_take: number; tau: number; tol: Tolerances }> };
  const byId = new Map(cat.signs.map((s) => [s.id, s]));
  const signs = new Map<string, SignRef>();
  for (const [id, r] of Object.entries(refs.signs)) {
    const info = byId.get(id);
    signs.set(id, {
      id, tier: info?.tier ?? 'watch', tau: r.tau, tol: r.tol, lookalikes: info?.lookalikes ?? [], shadowTake: r.shadow_take,
      takes: r.takes.map((t) => bindTake(segmentFromFlat(t), t.key_frame, params, t.source, t.take_id)),
    });
  }
  return {
    packId: m.pack_id, version: m.version, distribution: m.distribution, contentHash: m.content_hash, params,
    units: cat.units, catalogue: cat.signs, signs, selfcheck: m.selfcheck,
  };
}

/** Recompute the manifest's self-check. False means this engine and the pack builder disagree: don't use the pack. */
export function selfCheck(pack: ContentPack): { ok: boolean; expected: number; actual: number } {
  const c = pack.selfcheck;
  const take = pack.signs.get(c.sign_id)!.takes[c.take];
  let actual = 0;
  for (const v of take.descriptor.data) actual += v;
  return { ok: Math.abs(actual - c.descriptor_sum) <= c.tolerance, expected: c.descriptor_sum, actual };
}

export const lookalikesOf = (pack: ContentPack, id: string): SignRef[] =>
  pack.signs.get(id)!.lookalikes.map((x) => pack.signs.get(x)).filter((s): s is SignRef => !!s);
