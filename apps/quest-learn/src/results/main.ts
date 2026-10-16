// The model check page (dev/results.html): dictionary clips read by the web app's own pipeline, with frames.
import { SignReader, type Guess } from '../webcam/reader.js';
import { openLandmarkers, type Landmarkers } from '../webcam/vision.js';
import { clipStills, readClip } from './check.js';

interface Entry { sign: string; category: string; clip_id: string | null; v1: string | null }
interface Clip { id: string; names: string[]; kind: string; course: boolean }

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const KIND: Record<string, string> = { Alphabet: 'fingerspelling', Numbers: 'number', Phrases: 'phrase' };
const url = (clip: string) => `/dev/study/clip/${clip}`;
const pct = (n: number, d: number) => (d ? `${Math.round((100 * n) / d)}%` : '–');

let clips: Clip[] = [];
let reader: SignReader;
let marks: Landmarkers;
let stopping = false;

function seeded(seed: number) { // a fixed "random" sample, so a rerun checks the same clips
  return () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
}

function choose(set: string): Clip[] {
  const pick = (list: Clip[], n: number) => {
    const r = seeded(7), out = [...list];
    for (let i = out.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [out[i], out[j]] = [out[j], out[i]]; }
    return out.slice(0, n);
  };
  if (set === 'course') return clips.filter((c) => c.course);
  if (set === 'phrase' || set === 'word') return pick(clips.filter((c) => c.kind === set), 40);
  return clips.filter((c) => c.kind === set);
}

/** Right: the clip itself; also right: another recording carrying one of the clip's names. */
function judge(c: Clip, g: Guess): 'right' | 'other recording' | 'wrong' {
  if (g.sign.clip === c.id) return 'right';
  const mine = new Set(c.names.map((n) => n.toLowerCase().replace(/\s+\d+$/, '')));
  return [g.sign.name, ...g.sign.also].some((n) => mine.has(n.toLowerCase().replace(/\s+\d+$/, ''))) ? 'other recording' : 'wrong';
}

const totals = { n: 0, first: 0, top5: 0, sure: 0, sim: 0, simN: 0, time: 0 };
function tally(): void {
  $('t-first').textContent = pct(totals.first, totals.n);
  $('t-top5').textContent = pct(totals.top5, totals.n);
  $('t-sure').textContent = pct(totals.sure, totals.n);
  $('t-sim').textContent = totals.simN ? (totals.sim / totals.simN).toFixed(2) : '–';
  $('t-time').textContent = totals.n ? (totals.time / totals.n).toFixed(1) : '–';
}

async function row(c: Clip, mirror: boolean): Promise<void> {
  const t0 = performance.now();
  const r = await readClip(url(c.id), marks, reader, mirror);
  totals.n++;
  totals.time += (performance.now() - t0) / 1000;
  const guesses = r.reading?.guesses ?? [];
  const verdicts = guesses.map((g) => judge(c, g));
  const first = verdicts[0] && verdicts[0] !== 'wrong';
  const at = verdicts.findIndex((v) => v !== 'wrong');
  if (first) totals.first++;
  if (at >= 0) { totals.top5++; totals.sim += guesses[at].similarity; totals.simN++; }
  if (r.reading?.accepted) totals.sure++;
  tally();

  const el = document.createElement('div');
  el.className = `row${first ? '' : ' bad'}`;
  el.dataset.miss = first ? '' : '1';
  const verdict = !r.reading ? ['warn', 'No sign read (too little hand in view)']
    : first ? ['good', verdicts[0] === 'right' ? 'Read right first' : 'Read right first (another recording of it)']
      : at >= 0 ? ['warn', `Right sign ${at + 1}${at === 1 ? 'nd' : at === 2 ? 'rd' : 'th'}`] : ['bad', 'Missed'];
  el.innerHTML = `<div class="who"><b></b><div><span class="chip"></span></div>
      <div class="verdict ${verdict[0]}">${verdict[1]}</div>
      <div class="muted note">${r.reading ? (r.reading.accepted ? 'Stated' : 'Would say "not sure"') : ''}
      · hands in ${pct(r.handFrames, r.frames)} of ${r.frames} frames${r.readings.length > 1 ? ` · ${r.readings.length} attempts cut` : ''}</div></div>
    <div class="strip"></div><div class="guesses"></div>`;
  el.querySelector('.who b')!.textContent = c.names.slice(0, 3).join(' / ');
  el.querySelector('.chip')!.textContent = c.course ? `${c.kind}, course` : c.kind;
  el.querySelector('.strip')!.append(...r.thumbs);
  const box = el.querySelector('.guesses')!;
  guesses.slice(0, 3).forEach((g, k) => {
    const d = document.createElement('div');
    d.className = `guess${verdicts[k] !== 'wrong' ? ' right' : ''}`;
    d.innerHTML = `<div><b></b> <span class="muted"></span></div><div class="bar"><i></i></div><div class="strip"></div>`;
    d.querySelector('b')!.textContent = `${k + 1}. ${g.sign.name}`;
    d.querySelector('span')!.textContent = `${g.similarity.toFixed(2)}${verdicts[k] !== 'wrong' ? ' · this sign' : ''}`;
    (d.querySelector('.bar i') as HTMLElement).style.width = `${Math.max(0, Math.min(1, g.similarity)) * 100}%`;
    clipStills(url(g.sign.clip)).then((s) => d.querySelector('.strip')!.append(...s.map((x) => { const y = document.createElement('canvas'); Object.assign(y, { width: x.width, height: x.height }); y.getContext('2d')!.drawImage(x, 0, 0); return y; })));
    box.append(d);
  });
  el.hidden = $<HTMLInputElement>('misses').checked && first;
  $('rows').append(el);
}

async function run(): Promise<void> {
  const set = choose($<HTMLSelectElement>('set').value);
  const mirror = $<HTMLInputElement>('mirror').checked;
  Object.assign(totals, { n: 0, first: 0, top5: 0, sure: 0, sim: 0, simN: 0, time: 0 });
  tally();
  $('rows').replaceChildren();
  stopping = false;
  $<HTMLButtonElement>('go').disabled = true;
  $<HTMLButtonElement>('stop').disabled = false;
  for (let i = 0; i < set.length && !stopping; i++) {
    $('status').textContent = `Reading ${i + 1} of ${set.length}${mirror ? ', mirrored' : ''}…`;
    try { await row(set[i], mirror); } catch (err) { console.error(set[i].id, err); }
  }
  $('status').textContent = stopping ? 'Stopped.' : `Done: ${set.length} clips.`;
  $<HTMLButtonElement>('go').disabled = false;
  $<HTMLButtonElement>('stop').disabled = true;
}

async function start(): Promise<void> {
  const entries: Entry[] = await (await fetch('/dev/study/index.json')).json();
  const byClip = new Map<string, Clip>();
  for (const e of entries) {
    if (!e.clip_id) continue;
    const c = byClip.get(e.clip_id) ?? { id: e.clip_id, names: [], kind: KIND[e.category] ?? 'word', course: false };
    c.names.push(e.sign);
    if (e.v1) c.course = true;
    byClip.set(e.clip_id, c);
  }
  clips = [...byClip.values()];
  [reader, marks] = await Promise.all([SignReader.load(), openLandmarkers((t) => t && ($('status').textContent = t))]);
  $('run').textContent = `${reader.signCount} signs`;
  const meta = await (await fetch('/dev/recogniser/meta.json')).json();
  $('run').textContent = `run ${meta.run}, ${reader.signCount} signs`;
  $('status').textContent = `Ready: ${clips.filter((c) => c.course).length} course signs, ${clips.length} clips in all.`;
  $<HTMLButtonElement>('go').disabled = false;
}

$('go').addEventListener('click', () => void run());
$('stop').addEventListener('click', () => { stopping = true; });
$('misses').addEventListener('change', () => {
  const only = $<HTMLInputElement>('misses').checked;
  for (const el of $('rows').children) (el as HTMLElement).hidden = only && !(el as HTMLElement).dataset.miss;
});
start().catch((err) => { $('status').textContent = `Could not start: ${(err as Error).message}`; });
