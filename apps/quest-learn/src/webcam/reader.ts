// Reads whole signs from camera frames with the recogniser over every KSL Dictionary sign (`ashiria
// export-recogniser`): the engine's segmenter cuts each sign (hands up to hands down), the engine prepares it as
// the references were, and the ONNX model's embedding is matched against all of them. Runs in the browser; nothing
// leaves the device. Development only for now: the recogniser is served by the dev server.
import {
  AttemptSegmenter, FEATURE_DIM, TickClock, mirror, paramsFromJson, prepare, sequenceFeatures,
  type EngineParams, type Frame, type Segment,
} from '@ashiria/engine';

const BASE = '/dev/recogniser';
const ORT = '/node_modules/onnxruntime-web/dist';

export interface DictionarySign { clip: string; name: string; also: string[]; category: string }
export interface Guess { sign: DictionarySign; similarity: number }
/** One read sign: the best guesses, and whether the best is clear enough to state (otherwise "not sure"). */
export interface Reading { accepted: boolean; guesses: Guess[] }

interface Meta {
  frames: number; feature_dim: number; embedding_dim: number; engine: unknown; rest_height: number;
  accept: { sim: number; margin: number }; signs: DictionarySign[];
}

export class SignReader {
  readonly params: EngineParams;
  /** Mirror attempts first, for a left-handed signer (the references are right-handed). */
  leftHanded = false;
  /** Relaxed: accept a weaker match than the exported thresholds (set for 99.8% right on simulated attempts), for a
   * webcam, which sees hands less cleanly than the dictionary videos. */
  relaxed = true;
  private segmenter: AttemptSegmenter;
  private clock = new TickClock(30);
  private lastTime = -1;

  private constructor(private meta: Meta, private refs: Float32Array, private session: any, private ort: any) {
    const p = paramsFromJson(meta.engine);
    // Resting hands: the dictionary videos' height, a fair start for a webcam's view of a signer.
    this.params = { ...p, segmenter: { ...p.segmenter, rest_height: meta.rest_height } };
    this.segmenter = new AttemptSegmenter(this.params.segmenter);
    if (meta.feature_dim !== FEATURE_DIM) throw new Error(`Recogniser features ${meta.feature_dim} != engine ${FEATURE_DIM}`);
  }

  static async load(onStatus: (text: string) => void = () => {}): Promise<SignReader> {
    onStatus('Loading the sign recogniser…');
    const res = await fetch(`${BASE}/meta.json`);
    if (!res.ok) throw new Error('No recogniser yet: run `ashiria export-recogniser`.');
    const meta: Meta = await res.json();
    const refs = new Float32Array(await (await fetch(`${BASE}/refs.f32`)).arrayBuffer());
    const ort = await import(/* @vite-ignore */ `${ORT}/ort.wasm.min.mjs`);
    ort.env.wasm.wasmPaths = `${ORT}/`;
    ort.env.wasm.numThreads = 1; // threads need cross-origin isolation, which the dev server doesn't set
    const session = await ort.InferenceSession.create(`${BASE}/model.onnx`, { executionProviders: ['wasm'] });
    onStatus('');
    return new SignReader(meta, refs, session, ort);
  }

  get signCount(): number { return this.meta.signs.length; }

  /** The similarity a best guess needs, and its lead over the runner-up, to be stated rather than "not sure". */
  get acceptance(): { sim: number; margin: number } {
    const a = this.meta.accept;
    return this.relaxed ? { sim: Math.max(0.4, a.sim - 0.12), margin: 0.02 } : a;
  }
  /** idle (waiting for hands), active (signing), cooldown (just finished). */
  get state(): string { return this.segmenter.state; }

  /** Feed one camera frame (time in ms). Resolves to a reading when a sign has just finished, else null. */
  push(frame: Frame, timeMs: number): Promise<Reading | null> | null {
    const dt = this.lastTime < 0 ? 1 / 30 : (timeMs - this.lastTime) / 1000;
    this.lastTime = timeMs;
    let done: Segment | null = null;
    for (let k = this.clock.advance(dt); k > 0; k--) {
      const a = this.segmenter.push(frame);
      if (a) done = a.segment;
    }
    return done ? this.read(done) : null;
  }

  /** The recogniser's reading of one cut sign, or null if it has too little hand in it. */
  async read(seg: Segment): Promise<Reading | null> {
    const s = prepare(this.leftHanded ? mirror(seg) : seg, this.params, this.params.segmenter.rest_height);
    if (!s || s.length !== this.meta.frames) return null;
    const input = new this.ort.Tensor('float32', sequenceFeatures(s), [1, s.length, FEATURE_DIM]);
    const out = await this.session.run({ [this.session.inputNames[0]]: input });
    const e = out[this.session.outputNames[0]].data as Float32Array;
    const D = this.meta.embedding_dim, n = this.meta.signs.length;
    const sims = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      let d = 0;
      for (let k = 0; k < D; k++) d += this.refs[i * D + k] * e[k];
      sims[i] = d;
    }
    const top = Array.from(sims.keys()).sort((a, b) => sims[b] - sims[a]).slice(0, 5);
    const guesses = top.map((i) => ({ sign: this.meta.signs[i], similarity: sims[i] }));
    const { sim, margin } = this.acceptance;
    return { accepted: guesses[0].similarity >= sim && guesses[0].similarity - guesses[1].similarity >= margin, guesses };
  }

  reset(): void { this.segmenter.reset?.(); this.clock.reset(); this.lastTime = -1; }
}

/** Whether a dictionary sign is the lesson's sign (same name or alias, ignoring case and a trailing number). */
export function sameSign(sign: DictionarySign, gloss: string, aliases: string[] = []): boolean {
  const norm = (s: string) => s.toLowerCase().replace(/\s+\d+$/, '').trim();
  const want = new Set([gloss, ...aliases].map(norm));
  return [sign.name, ...sign.also].some((n) => want.has(norm(n)));
}
