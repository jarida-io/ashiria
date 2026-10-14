// Generates the app icons in public/icons from one SVG: `node scripts/icons.mjs`.
// A white open hand on a teal gradient, full-bleed so it also works as the maskable icon.
import { mkdir } from 'node:fs/promises';
import sharp from 'sharp';

const finger = (x, y, h, r) => `<rect x="${x}" y="${y}" width="44" height="${h}" rx="22" transform="rotate(${r} ${x + 22} ${y + h})"/>`;
const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512">
  <defs>
    <linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#1fb5c9"/>
      <stop offset="100%" stop-color="#0b3a5c"/>
    </linearGradient>
  </defs>
  <rect width="512" height="512" fill="url(#g)"/>
  <g fill="#ffffff">
    <rect x="166" y="236" width="184" height="150" rx="62"/>
    ${finger(170, 136, 130, -8)}
    ${finger(222, 108, 150, -2)}
    ${finger(274, 116, 142, 4)}
    ${finger(322, 150, 112, 10)}
    ${finger(150, 236, 120, -38)}
  </g>
  <circle cx="400" cy="120" r="22" fill="#ffd166"/>
</svg>`;

await mkdir('public/icons', { recursive: true });
const png = (size, out) => sharp(Buffer.from(svg)).resize(size, size).png().toFile(out);
await png(512, 'public/icons/icon-512.png');
await png(192, 'public/icons/icon-192.png');
await png(512, 'public/icons/icon-512-maskable.png');
console.log('icons written to public/icons');
