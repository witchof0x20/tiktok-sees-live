// Minimal headless-Chromium driver: opens a URL with WebGPU enabled and evaluates an expression once it's truthy.
// usage: node tools/cdp.mjs <url> <expr-to-await> [timeoutMs]
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
const [url, expr, timeout = '120000'] = process.argv.slice(2);
const prof = fs.mkdtempSync(path.join(process.env.TMPDIR || os.tmpdir(), 'cdp-'));
const flags = (process.env.CHROME_FLAGS || '--enable-unsafe-webgpu --enable-features=Vulkan,WebGPUService --use-angle=vulkan --ignore-gpu-blocklist').split(' ');
const chrome = spawn(process.env.CHROME || 'chromium', ['--headless=new', '--remote-debugging-port=0', `--user-data-dir=${prof}`, '--no-first-run', ...flags, 'about:blank'], { stdio: ['ignore', 'ignore', 'pipe'] });
const wsUrl = await new Promise((res, rej) => { let buf = ''; chrome.stderr.on('data', d => { buf += d; const m = buf.match(/DevTools listening on (ws:\S+)/); if (m) res(m[1]); }); setTimeout(() => rej(new Error('no devtools\n' + buf)), 20000); });
const browser = new WebSocket(wsUrl);
await new Promise(r => browser.onopen = r);
let id = 0; const pending = new Map();
browser.onmessage = e => { const m = JSON.parse(e.data); if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); } else if (m.method === 'Runtime.consoleAPICalled' && process.env.CONSOLE) console.error('[console]', m.params.args.map(a => a.value ?? a.description).join(' ')); };
const send = (method, params = {}, sessionId) => new Promise(r => { const i = ++id; pending.set(i, r); browser.send(JSON.stringify({ id: i, method, params, sessionId })); });
const { result: { targetId } } = await send('Target.createTarget', { url: 'about:blank' });
const { result: { sessionId } } = await send('Target.attachToTarget', { targetId, flatten: true });
await send('Runtime.enable', {}, sessionId);
// VIEWPORT=WxHxDPR emulates a phone-sized, touch, mobile viewport.
if (process.env.VIEWPORT) {
  const [width, height, dpr] = process.env.VIEWPORT.split('x').map(Number);
  await send('Emulation.setDeviceMetricsOverride', { width, height, deviceScaleFactor: dpr || 1, mobile: true }, sessionId);
  await send('Emulation.setTouchEmulationEnabled', { enabled: true }, sessionId);
}
await send('Page.navigate', { url }, sessionId);
const t0 = Date.now(); let out;
while (Date.now() - t0 < +timeout) {
  const r = await send('Runtime.evaluate', { expression: expr, awaitPromise: true, returnByValue: true }, sessionId);
  out = r.result?.result?.value;
  if (out) break;
  await new Promise(r => setTimeout(r, 500));
}
console.log(typeof out === 'string' ? out : JSON.stringify(out, null, 1));
// SCREENSHOT=path.png saves the viewport after the expression resolves.
if (process.env.SCREENSHOT) {
  const { result } = await send('Page.captureScreenshot', { format: 'png' }, sessionId);
  fs.writeFileSync(process.env.SCREENSHOT, Buffer.from(result.data, 'base64'));
}
chrome.kill();
await new Promise(r => chrome.once('exit', r));
fs.rmSync(prof, { recursive: true, force: true, maxRetries: 5 });
process.exit(0);
