/**
 * The 2D landing page judges and learners see when they open the link: what they need, a big Start button,
 * Start over (for shared headsets), and the privacy policy. It hides while the immersive session runs.
 */
import { VisibilityState, type World } from '@iwsdk/core';
import { app } from './app-state.js';

export function setupLanding(world: World): void {
  const landing = document.getElementById('landing');
  const start = document.getElementById('start') as HTMLButtonElement | null;
  const note = document.getElementById('note');
  const over = document.getElementById('start-over');
  if (!landing || !start || !note || !over) return;

  const nav = navigator as Navigator & { xr?: { isSessionSupported?: (mode: string) => Promise<boolean> } };
  const ready = (ok: boolean) => {
    start.disabled = !ok;
    start.textContent = ok ? 'Start' : 'Open this link on a Meta Quest';
    note.textContent = ok ? '' : 'This page needs a Meta Quest 3 or 3S: open the same link in the Meta Quest Browser.';
  };
  if (!world.xrEnabled || !nav.xr?.isSessionSupported) ready(false);
  else nav.xr.isSessionSupported('immersive-ar').then(ready, () => ready(false));

  start.addEventListener('click', () => world.launchXR());
  over.addEventListener('click', () => {
    if (confirm('Start over? This clears your saved progress and settings on this device.')) {
      app.deleteMyData();
      note.textContent = 'Cleared. Press Start to begin from the first sign.';
    }
  });
  world.visibilityState.subscribe((v) => { landing.hidden = v !== VisibilityState.NonImmersive; });
}
