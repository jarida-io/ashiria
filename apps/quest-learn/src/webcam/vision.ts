// The camera and MediaPipe's hand and pose landmarkers, in the browser: the same models that read the dictionary
// videos (training/models/mediapipe), so a webcam signer is seen the way the references were. Frames stay on the
// device; nothing is uploaded. Development only for now: the models and MediaPipe's runtime come from the dev server.
import type { CameraHand, Point } from '@ashiria/engine';
import { importFresh } from './load.js';

const MEDIAPIPE = '/node_modules/@mediapipe/tasks-vision';
const MODELS = '/dev/mediapipe';

export interface CameraFrame { hands: CameraHand[]; pose: { image: Point[]; world: Point[] } | null; time: number }

interface Landmarker { detectForVideo(video: HTMLVideoElement, ms: number): any; close(): void }

export interface Vision {
  video: HTMLVideoElement;
  /** True when frames are flipped before tracking, for a camera that delivers a mirror image (selfie-style). */
  readonly mirrored: boolean;
  setMirrored(on: boolean): void;
  /** Calls `onFrame` with each new camera frame's landmarks until `stop`. */
  run(onFrame: (f: CameraFrame) => void): void;
  stop(): void;
}

export interface Landmarkers {
  /** Hands and body in one frame (`ms` must increase from call to call). */
  detect(source: HTMLVideoElement | HTMLCanvasElement, ms: number): CameraFrame;
  close(): void;
}

/** MediaPipe's hand and pose landmarkers (about 17 MB, cached after the first time), set as in
 * training/configs/default.yaml (extract) so what they see matches the references. */
export async function openLandmarkers(onStatus: (text: string) => void = () => {}): Promise<Landmarkers> {
  onStatus('Loading hand tracking…');
  const mp = await importFresh(`${MEDIAPIPE}/vision_bundle.mjs`);
  const files = await mp.FilesetResolver.forVisionTasks(`${MEDIAPIPE}/wasm`);
  const create = async (Task: any, options: any): Promise<Landmarker> => {
    try {
      return await Task.createFromOptions(files, { ...options, baseOptions: { ...options.baseOptions, delegate: 'GPU' } });
    } catch {
      return await Task.createFromOptions(files, { ...options, baseOptions: { ...options.baseOptions, delegate: 'CPU' } });
    }
  };
  const hands = await create(mp.HandLandmarker, {
    baseOptions: { modelAssetPath: `${MODELS}/hand_landmarker.task` }, runningMode: 'VIDEO', numHands: 2,
    minHandDetectionConfidence: 0.5, minHandPresenceConfidence: 0.5, minTrackingConfidence: 0.5,
  });
  const pose = await create(mp.PoseLandmarker, {
    baseOptions: { modelAssetPath: `${MODELS}/pose_landmarker_full.task` }, runningMode: 'VIDEO', numPoses: 1,
  });
  onStatus('');
  return {
    detect(source, ms) {
      const h = hands.detectForVideo(source as HTMLVideoElement, ms);
      const p = pose.detectForVideo(source as HTMLVideoElement, ms);
      const labels = h.handedness ?? h.handednesses ?? [];
      return {
        hands: h.landmarks.map((image: Point[], k: number) => ({ image, world: h.worldLandmarks[k], label: labels[k]?.[0]?.categoryName ?? '' })),
        pose: p.landmarks?.length && p.worldLandmarks?.length ? { image: p.landmarks[0], world: p.worldLandmarks[0] } : null,
        time: ms,
      };
    },
    close() { hands.close(); pose.close(); },
  };
}

/** Opens the camera (the browser asks for permission) and MediaPipe's landmarkers. */
export async function openVision(onStatus: (text: string) => void = () => {}): Promise<Vision> {
  onStatus('Asking for the camera…');
  const stream = await navigator.mediaDevices.getUserMedia({
    // 720p if the camera has it: MediaPipe crops each hand from the frame, so more pixels mean steadier fingers.
    video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 } }, audio: false,
  });
  const video = document.createElement('video');
  Object.assign(video, { srcObject: stream, muted: true, playsInline: true });
  await video.play();
  const marks = await openLandmarkers(onStatus);

  let running = false, last = -1, mirrored = false;
  // A mirrored camera's frames are flipped back before tracking, so MediaPipe sees the signer as on video.
  const flip = document.createElement('canvas');
  const flipCtx = flip.getContext('2d')!;
  const source = (): HTMLVideoElement | HTMLCanvasElement => {
    if (!mirrored) return video;
    if (flip.width !== video.videoWidth) { flip.width = video.videoWidth; flip.height = video.videoHeight; }
    flipCtx.setTransform(-1, 0, 0, 1, flip.width, 0);
    flipCtx.drawImage(video, 0, 0);
    return flip;
  };
  const step = () => {
    if (!running) return;
    if (video.readyState >= 2) {
      const ms = Math.max(performance.now(), last + 1); // MediaPipe needs increasing timestamps
      if (ms !== last) {
        last = ms;
        frameHandler?.(marks.detect(source(), ms));
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
    get mirrored() { return mirrored; },
    setMirrored(on) { mirrored = on; },
    run(onFrame) { frameHandler = onFrame; if (!running) { running = true; schedule(); } },
    stop() {
      running = false;
      stream.getTracks().forEach((t) => t.stop());
      marks.close();
    },
  };
}
