// A plain-HTTP door to the dev server, for desktop previews such as the Claude app's browser pane, which refuses
// IWSDK's self-signed certificate. http://localhost:8082 forwards every request (and WebSocket, for live reload) to
// https://localhost:8081. http://localhost is a secure context, so WebXR and the headset emulator still work.
// Run the dev server first (npm run dev), then: node scripts/http-preview.mjs
import http from 'node:http';
import https from 'node:https';
import tls from 'node:tls';

const PORT = 8082, TARGET = { host: 'localhost', port: 8081 };

const server = http.createServer((req, res) => {
  const headers = { ...req.headers, host: `${TARGET.host}:${TARGET.port}` };
  const upstream = https.request({ ...TARGET, method: req.method, path: req.url, headers, rejectUnauthorized: false }, (r) => {
    res.writeHead(r.statusCode ?? 502, r.headers);
    r.pipe(res);
  });
  upstream.on('error', (err) => {
    res.writeHead(502, { 'content-type': 'text/plain' });
    res.end(`The dev server on https://localhost:${TARGET.port} isn't answering (${err.code ?? err.message}). Start it with npm run dev.`);
  });
  req.pipe(upstream);
});

server.on('upgrade', (req, socket, head) => {
  const up = tls.connect({ ...TARGET, rejectUnauthorized: false, servername: TARGET.host }, () => {
    const lines = [`${req.method} ${req.url} HTTP/1.1`, ...Object.entries({ ...req.headers, host: `${TARGET.host}:${TARGET.port}` }).map(([k, v]) => `${k}: ${v}`)];
    up.write(lines.join('\r\n') + '\r\n\r\n');
    if (head.length) up.write(head);
    socket.pipe(up).pipe(socket);
  });
  const close = () => { socket.destroy(); up.destroy(); };
  up.on('error', close);
  socket.on('error', close);
});

server.listen(PORT, 'localhost', () => console.log(`http://localhost:${PORT} -> https://localhost:${TARGET.port}`));
