// Classic worker running one MediaPipe FaceLandmarker. Several of these take camera frames round-robin so face
// finding (the slowest stage by far) runs in parallel instead of blocking the main thread.
//
// The vendored bundle is an ES module, but MediaPipe's wasm loader needs importScripts, which module workers lack.
// The bundle's only module syntax is its trailing export list, so it's rewritten into a global and loaded classic.
let finder = null;

async function init({ vendor, delegate, numFaces }) {
  const src = await (await fetch(vendor + 'vision_bundle.js')).text();
  const m = src.match(/export\s*\{([^}]*)\}\s*;?\s*$/);
  if (!m || src.slice(0, m.index).includes('import.meta')) throw new Error('unexpected vision_bundle.js layout');
  const names = m[1].split(',').map(p => p.trim().split(/\s+as\s+/)).map(([a, b]) => `${JSON.stringify(b ?? a)}: ${a}`);
  const url = URL.createObjectURL(new Blob([src.slice(0, m.index) + `\nself.MPVision = {${names.join(', ')}};\n`], { type: 'text/javascript' }));
  importScripts(url);
  URL.revokeObjectURL(url);
  const fileset = { wasmLoaderPath: vendor + 'vision_wasm_internal.js', wasmBinaryPath: vendor + 'vision_wasm_internal.wasm' };
  const make = d => self.MPVision.FaceLandmarker.createFromOptions(fileset, {
    ...(d === 'GPU' ? { canvas: new OffscreenCanvas(1, 1) } : {}),
    baseOptions: { modelAssetPath: vendor + 'face_landmarker.task', delegate: d }, runningMode: 'VIDEO', numFaces,
    minFaceDetectionConfidence: 0.5, minFacePresenceConfidence: 0.6, minTrackingConfidence: 0.6,
  });
  // WebGL where it works, else MediaPipe's CPU (WASM) path.
  try { finder = await make(delegate); self.delegate = delegate; }
  catch (err) { if (delegate === 'CPU') throw err; finder = await make('CPU'); self.delegate = 'CPU'; }
}

self.onmessage = async ({ data }) => {
  if (data.type === 'init') {
    try { await init(data); postMessage({ type: 'ready', delegate: self.delegate }); }
    catch (err) { postMessage({ type: 'error', message: String(err?.message || err) }); }
    return;
  }
  if (data.type === 'frame') {
    const t0 = performance.now();
    let faces = [];
    try {
      const r = finder.detectForVideo(data.bitmap, data.ts);
      faces = r.faceLandmarks.map(L => { const a = new Float32Array(L.length * 2); L.forEach((p, i) => { a[2 * i] = p.x; a[2 * i + 1] = p.y; }); return a; });
    } finally { data.bitmap.close(); }
    postMessage({ type: 'result', id: data.id, faces, ms: performance.now() - t0 }, faces.map(a => a.buffer));
  }
};
