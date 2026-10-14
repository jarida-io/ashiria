/**
 * Ashiria Learn: practise Kenyan Sign Language with hand tracking (WebXR, Immersive Web SDK).
 */
import { World } from '@iwsdk/core';
import projectOptions from 'virtual:iwsdk-project';
import { app } from './app-state.js';
import { loadPack } from './content.js';
import { HandSamplerSystem } from './hand-sampler.js';
import { LessonSystem } from './lesson.js';
import { ShadowHandsSystem } from './shadow-hands.js';

if (import.meta.env.PROD && 'serviceWorker' in navigator) {
  navigator.serviceWorker.register(`${import.meta.env.BASE_URL}sw.js`).catch(() => { /* still works online */ });
}

World.create(document.getElementById('scene-container') as HTMLDivElement, projectOptions).then((world) => {
  world.registerSystem(HandSamplerSystem);
  world.registerSystem(ShadowHandsSystem);
  world.registerSystem(LessonSystem);
  // Development builds only: handles for scripted tests in the IWSDK emulator.
  if (import.meta.env.DEV) Object.assign(window, { __ashiria: { world, app, HandSamplerSystem, LessonSystem } });
  // Installed app (packaged PWA on Quest): tapping the icon is the user's go-ahead, so enter mixed reality
  // straight away. Only the installed app has getDigitalGoodsService; a browser tab keeps its Start button.
  const nav = navigator as Navigator & { xr?: { isSessionSupported?: (mode: string) => Promise<boolean> } };
  if ('getDigitalGoodsService' in window && nav.xr?.isSessionSupported) {
    nav.xr.isSessionSupported('immersive-ar').then((ok) => { if (ok) world.launchXR(); }).catch(() => {});
  }
  loadPack().then(
    (pack) => { app.pack = pack; app.goTo(0); },
    (err) => { app.packError = String(err instanceof Error ? err.message : err); app.changed(); },
  );
});
