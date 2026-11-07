"""Merges autotuner outputs (each {opt, nets, totalUs, adapter}) into one tuning file: later files' nets win."""
import json, sys

out, files = sys.argv[1], sys.argv[2:]
merged = {'opt': None, 'nets': {}, 'totalUs': {}}
for f in files:
    d = json.load(open(f))
    if 'error' in d:
        raise SystemExit(f'{f}: {d["error"]}')
    if merged['opt'] and merged['opt'] != d['opt']:
        raise SystemExit(f'{f}: tuning options differ ({merged["opt"]} vs {d["opt"]})')
    merged['opt'] = d['opt']
    merged['adapter'] = d.get('adapter', merged.get('adapter'))
    merged['nets'].update(d['nets'])
    for n, groups in d['nets'].items():
        merged['totalUs'][n] = round(sum(g['us'] for g in groups), 1)
json.dump(merged, open(out, 'w'), indent=1)
print(out, merged['totalUs'])
