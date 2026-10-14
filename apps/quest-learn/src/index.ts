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

World.create(document.getElementById('scene-container') as HTMLDivElement, projectOptions).then((world) => {
  world.registerSystem(HandSamplerSystem);
  world.registerSystem(ShadowHandsSystem);
  world.registerSystem(LessonSystem);
  loadPack().then(
    (pack) => { app.pack = pack; app.changed(); },
    (err) => { app.packError = String(err instanceof Error ? err.message : err); app.changed(); },
  );
});
