"""Prints a TFLite graph with DEQUANTIZE folded away: op, attrs, input/output shapes."""
import sys, struct
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from tflite import read

m = read(sys.argv[1])
T = m['tensors']
deq = {o['outputs'][0]: o['inputs'][0] for o in m['ops'] if o['op'] == 'DEQUANTIZE'}
name = lambda i: T[deq.get(i, i)]['name'][-38:]
for o in m['ops']:
    if o['op'] == 'DEQUANTIZE':
        continue
    ins = []
    for i in o['inputs']:
        if i < 0:
            ins.append('-'); continue
        t = T[deq.get(i, i)]
        const = bool(t['data'])
        if const and t['type'] == 'int32':
            ins.append('c' + str(list(struct.unpack('<%di' % (len(t['data']) // 4), t['data']))))
        else:
            ins.append(('c' if const else '') + str(T[i]['shape']))
    outs = [str(T[i]['shape']) for i in o['outputs']]
    opts = {k: v for k, v in o['opts'].items() if v not in (None, (1, 1)) and not (k == 'mult' and v == 1)}
    print(f"{o['op']:18} {str(opts):60} {' '.join(ins):60} -> {' '.join(outs)}")
