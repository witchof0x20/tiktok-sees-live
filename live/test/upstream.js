// Some checks compare against ayoships.com's own reference code and sample photo, which this repo doesn't
// redistribute. tools/fetch-upstream.sh downloads them into upstream/ for local runs.
export const UP = new URL('../../upstream/tools/tiktok-face/', import.meta.url).href;
export const NETS = new URL('../nets/', import.meta.url).href;
export const VENDOR = new URL('../vendor/', import.meta.url).href;

export async function requireUpstream(log) {
  const ok = await fetch(UP + 'engine.js', { method: 'HEAD' }).then(r => r.ok, () => false);
  if (!ok) log('This check needs the original tool\'s reference files. Run tools/fetch-upstream.sh, then serve the repo root locally.');
  return ok;
}
// Loads upstream's classic engine.js (defines self.FaceNet).
export function loadReferenceEngine() {
  return new Promise((resolve, reject) => {
    if (self.FaceNet) return resolve(self.FaceNet);
    const s = document.createElement('script');
    s.src = UP + 'engine.js'; s.onload = () => resolve(self.FaceNet); s.onerror = reject;
    document.head.append(s);
  });
}
