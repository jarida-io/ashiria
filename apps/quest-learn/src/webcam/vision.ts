// The camera and MediaPipe's hand and pose landmarkers, in the browser: the same models that read the dictionary
// videos (training/models/mediapipe), so a webcam signer is seen the way the references were. Frames stay on the
// device; nothing is uploaded. Development only for now: the models and MediaPipe's runtime come from the dev server.
import type { CameraHand, Point } from '@ashiria/engine';

const MEDIAPIPE = '/node_modules/@mediapipe/tasks-vision';
const MODELS = '/dev/mediapipe';

export interface CameraFrame { hands: CameraHand[]; pose: { image: Point[]; world: Point[] } | null; time: number }

interface Landmarker { detectForVideo(video: HTMLVideoElement, ms: number): any; close(): void }

export interface Vision {
  video: HTMLVideoElement;
  /** Calls `onFrame` with each new camera frame's landmarks until `stop`. */
  run(onFrame: (f: CameraFrame) => void): void;
  stop(): void;
}

/** Opens the camera (the browser asks for permission) and loads MediaPipe (about 17 MB, cached after the first time). */
export async function openVision(onStatus: (text: string) => void = () => {}): Promise<Vision> {
  onStatus('Asking for the camera…');
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30 } }, audio: false,
  });
  const video = document.createElement('video');
  Object.assign(video, { srcObject: stream, muted: true, playsInline: true });
  await video.play();

  onStatus('Loading hand tracking…');
  const mp = await import(/* @vite-ignore */ `${MEDIAPIPE}/vision_bundle.mjs`);
  const files = await mp.FilesetResolver.forVisionTasks(`${MEDIAPIPE}/wasm`);
  const create = async (Task: any, options: any): Promise<Landmarker> => {
    try {
      return await Task.createFromOptions(files, { ...options, baseOptions: { ...options.baseOptions, delegate: 'GPU' } });
    } catch {
      return await Task.createFromOptions(files, { ...options, baseOptions: { ...options.baseOptions, delegate: 'CPU' } });
    }
  };
  // Settings as in training/configs/default.yaml (extract), so live tracking matches the references.
  const hands = await create(mp.HandLandmarker, {
    baseOptions: { modelAssetPath: `${MODELS}/hand_landmarker.task` }, runningMode: 'VIDEO', numHands: 2,
    minHandDetectionConfidence: 0.5, minHandPresenceConfidence: 0.5, minTrackingConfidence: 0.5,
  });
  const pose = await create(mp.PoseLandmarker, {
    baseOptions: { modelAssetPath: `${MODELS}/pose_landmarker_full.task` }, runningMode: 'VIDEO', numPoses: 1,
  });
  onStatus('');

  let running = false, last = -1;
  const step = () => {
    if (!running) return;
    if (video.readyState >= 2) {
      const ms = Math.max(performance.now(), last + 1); // MediaPipe needs increasing timestamps
      if (ms !== last) {
        last = ms;
        const h = hands.detectForVideo(video, ms);
        const p = pose.detectForVideo(video, ms);
        const labels = h.handedness ?? h.handednesses ?? [];
        frameHandler?.({
          hands: h.landmarks.map((image: Point[], k: number) => ({ image, world: h.worldLandmarks[k], label: labels[k]?.[0]?.categoryName ?? '' })),
          pose: p.landmarks?.length && p.worldLandmarks?.length ? { image: p.landmarks[0], world: p.worldLandmarks[0] } : null,
          time: ms,
        });
      }
    }
    schedule();
  };
  const schedule = () => {
    if ('requestVideoFrameCallback' in video) (video as any).requestVideoFrameCallback(step);
    else requestAnimationFrame(step);
  };
  let frameHandler: ((f: CameraFrame) => void) | null = null;

  return {
    video,
    run(onFrame) { frameHandler = onFrame; if (!running) { running = true; schedule(); } },
    stop() {
      running = false;
      stream.getTracks().forEach((t) => t.stop());
      hands.close();
      pose.close();
    },
  };
}
