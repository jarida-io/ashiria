/**
 * Asset catalog shared by the runtime and the editor (deterministic, side-effect free).
 */
import { AssetType, defineAssets } from '@iwsdk/core';

const publicAssetUrl = (filePath: string): string => `${import.meta.env.BASE_URL}${filePath.replace(/^\/+/u, '')}`;

export default defineAssets({
  'lesson-panel': {
    url: publicAssetUrl('ui/lesson.uikitml'),
    type: AssetType.UIKitML,
    name: 'Lesson Panel',
  },
});
