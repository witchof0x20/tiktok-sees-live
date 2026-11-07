"""Converts a MediaPipe face .tflite into the engine's float graph format: <name>.json + <name>.bin (float32).

Done offline so the browser never parses flatbuffers. float16 weights (DEQUANTIZE) are widened to float32
losslessly; RELU / PReLU are folded into their producer when it has no other consumer; channel PADs become
'cpad'; RESHAPE / CONCATENATION / LOGISTIC around the heads are dropped (the GPU post-processing reads the head
convs directly). Weights keep TFLite layouts: conv OHWI, depthwise 1HWC.
usage: convert_tflite.py <model.tflite> <out_prefix> <name> <input_range_min> <input_range_max>
"""
import json, struct, sys
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from tflite import read


def f16(b):
    return list(struct.unpack('<%de' % (len(b) // 2), b))


def f32(b):
    return list(struct.unpack('<%df' % (len(b) // 4), b))


def same_pad(n, k, s):
    out = (n + s - 1) // s
    total = max((out - 1) * s + k - n, 0)
    return out, total // 2


def main(path, prefix, name, lo, hi):
    m = read(path)
    T = m['tensors']
    const = {}
    for i, t in enumerate(T):
        if t['data']:
            const[i] = f16(t['data']) if t['type'] == 'float16' else f32(t['data']) if t['type'] == 'float32' else None
    ops = []
    for o in m['ops']:
        if o['op'] == 'DEQUANTIZE':
            const[o['outputs'][0]] = const[o['inputs'][0]]
        else:
            ops.append(o)
    consumers = {}
    for k, o in enumerate(ops):
        for i in o['inputs']:
            consumers.setdefault(i, []).append(k)

    blob = []
    def put(vals):
        off = len(blob)
        blob.extend(vals)
        return off

    nodes, alias, dropped = [], {}, set()
    inp = m['inputs'][0]
    _, H, W, C = T[inp]['shape']
    tname = lambda i: alias.get(i, f't{i}')
    nodes.append(dict(name=tname(inp), op='data', C=C, H=H, W=W, range=[lo, hi]))
    shape = {inp: (H, W, C)}

    for k, o in enumerate(ops):
        if k in dropped:
            continue
        op, ins, out = o['op'], o['inputs'], o['outputs'][0]
        oshape = T[out]['shape']
        if op in ('CONV_2D', 'DEPTHWISE_CONV_2D'):
            x = ins[0]
            h, w, c = shape[x]
            wshape = T[ins[1]]['shape']
            if op == 'CONV_2D':
                co, kh, kw, ci = wshape
                assert ci == c, (ci, c)
            else:
                _, kh, kw, co = wshape
                assert co == c and o['opts']['mult'] == 1
            assert o['opts']['dilation'] == (1, 1)
            sh, sw = o['opts']['stride']
            assert sh == sw
            if o['opts']['padding'] == 'same':
                oh, ph = same_pad(h, kh, sh)
                ow, pw = same_pad(w, kw, sw)
            else:
                oh, ph, ow, pw = (h - kh) // sh + 1, 0, (w - kw) // sw + 1, 0
            assert ph == pw and [1, oh, ow, co] == list(oshape[:3]) + [co] or oshape[-1] == co, (oshape, oh, ow, co)
            node = dict(name=tname(out), op='conv' if op == 'CONV_2D' else 'dw', ins=[tname(x)], C=co, H=oh, W=ow,
                        k=[kh, kw], s=sh, p=ph, act=o['opts']['act'], wOff=put(const[ins[1]]), bOff=put(const[ins[2]]))
            assert node['act'] in (None, 'relu')
            # Fold a following RELU / PRELU that is this tensor's only consumer.
            nxt = consumers.get(out, [])
            if len(nxt) == 1 and ops[nxt[0]]['op'] in ('RELU', 'PRELU') and node['act'] is None:
                f = ops[nxt[0]]
                node['act'] = 'relu' if f['op'] == 'RELU' else 'prelu'
                if f['op'] == 'PRELU':
                    node['aOff'] = put(const[f['inputs'][1]])
                dropped.add(nxt[0])
                alias[f['outputs'][0]] = node['name']
                out = f['outputs'][0]
            nodes.append(node)
            shape[out] = (oh, ow, co)
        elif op == 'ADD':
            a, b = ins
            assert shape[a] == shape[b]
            node = dict(name=tname(out), op='add', ins=[tname(a), tname(b)], C=shape[a][2], H=shape[a][0], W=shape[a][1], act=o['opts']['act'])
            nxt = consumers.get(out, [])
            if len(nxt) == 1 and ops[nxt[0]]['op'] in ('RELU', 'PRELU') and node['act'] is None:
                f = ops[nxt[0]]
                node['act'] = 'relu' if f['op'] == 'RELU' else 'prelu'
                if f['op'] == 'PRELU':
                    node['aOff'] = put(const[f['inputs'][1]])
                dropped.add(nxt[0])
                alias[f['outputs'][0]] = node['name']
                out = f['outputs'][0]
            nodes.append(node)
            shape[out] = shape[a]
        elif op in ('RELU', 'PRELU'):
            x = ins[0]
            node = dict(name=tname(out), op='act', ins=[tname(x)], C=shape[x][2], H=shape[x][0], W=shape[x][1],
                        act='relu' if op == 'RELU' else 'prelu')
            if op == 'PRELU':
                node['aOff'] = put(const[ins[1]])
            nodes.append(node)
            shape[out] = shape[x]
        elif op == 'MAX_POOL_2D':
            x = ins[0]
            h, w, c = shape[x]
            fh, fw = o['opts']['filter']
            s = o['opts']['stride'][0]
            if o['opts']['padding'] == 'same':
                oh, ph = same_pad(h, fh, s)
                assert ph == 0 and (oh - 1) * s + fh <= h, 'max pool with real padding'
            else:
                oh = (h - fh) // s + 1
            node = dict(name=tname(out), op='pool', ins=[tname(x)], C=c, H=oh, W=oh, k=[fh, fw], s=s, p=0)
            nodes.append(node)
            shape[out] = (oh, oh, c)
        elif op == 'PAD':
            x = ins[0]
            pads = struct.unpack('<8i', T[ins[1]]['data'])
            assert pads[:7] == (0,) * 7, pads
            h, w, c = shape[x]
            nodes.append(dict(name=tname(out), op='cpad', ins=[tname(x)], C=c + pads[7], H=h, W=w))
            shape[out] = (h, w, c + pads[7])
        elif op in ('RESHAPE', 'CONCATENATION', 'LOGISTIC'):
            continue
        else:
            raise ValueError('unsupported op ' + op)

    # Heads: the convs feeding RESHAPE / LOGISTIC / graph outputs.
    produced = {n['name'] for n in nodes}
    used = {i for n in nodes for i in n.get('ins', [])}
    heads = [n['name'] for n in nodes if n['name'] not in used and n['op'] != 'data']
    json.dump(dict(name=name, nodes=nodes, heads=heads), open(prefix + '.json', 'w'), indent=0)
    open(prefix + '.bin', 'wb').write(struct.pack('<%df' % len(blob), *blob))
    print(name, len(nodes), 'nodes', len(blob), 'floats; heads', [(h, next(n for n in nodes if n['name'] == h)['C']) for h in heads])


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]), float(sys.argv[5]))
