"""Adds ?v=<version> to relative .js/.css/.wasm references in our own files, so a deploy loads as one consistent
version even with GitHub Pages' 10-minute caching. Vendored files are left alone.
usage: stamp.py <site_dir> <version>"""
import pathlib, re, sys

site, version = pathlib.Path(sys.argv[1]), sys.argv[2]
if not re.fullmatch(r'[0-9a-f]{6,40}', version):
    raise SystemExit('version must be a hex commit id')
ref = re.compile(r'''(['"])((?:\.{1,2}/|[\w-]+/)*[\w.-]+\.(?:js|css|wasm))\1''')
count = 0
for path in sorted(site.rglob('*')):
    if path.suffix not in ('.js', '.html') or 'vendor' in path.parts:
        continue
    text = path.read_text()
    new, n = ref.subn(lambda m: f'{m[1]}{m[2]}?v={version}{m[1]}', text)
    if n:
        path.write_text(new)
        count += n
print(f'stamped {count} references with v={version}')
