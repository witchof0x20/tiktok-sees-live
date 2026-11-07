#!/bin/sh
# Downloads ayoships.com/tools/tiktok-face into upstream/ (git-ignored) for local use: the original tool itself and
# the reference engine, face.js and sample photo used by some test pages. Not redistributed in this repo.
set -eu
cd "$(dirname "$0")/.."
base=https://ayoships.com
for f in styles.css app.js ayo.jpg \
  tools/tiktok-face/index.html tools/tiktok-face/tool.js tools/tiktok-face/tool.css tools/tiktok-face/face.js \
  tools/tiktok-face/engine.js tools/tiktok-face/sample.jpg tools/tiktok-face/instrument.woff2 \
  tools/tiktok-face/expnet.json tools/tiktok-face/expnet.bin tools/tiktok-face/agenet.json tools/tiktok-face/agenet.bin \
  tools/tiktok-face/vendor/vision_bundle.js tools/tiktok-face/vendor/vision_wasm_internal.js \
  tools/tiktok-face/vendor/vision_wasm_internal.wasm tools/tiktok-face/vendor/face_landmarker.task; do
  mkdir -p "upstream/$(dirname "$f")"
  curl -sSfL "$base/$f" -o "upstream/$f"
  echo "upstream/$f"
done
