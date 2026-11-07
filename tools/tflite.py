"""Minimal, dependency-free TFLite flatbuffer reader (only the parts the face models use).

Field indices follow tensorflow/lite/schema/schema.fbs. Every offset is bounds-checked; anything unexpected raises.
"""
import struct

OPS = {0: 'ADD', 1: 'AVERAGE_POOL_2D', 2: 'CONCATENATION', 3: 'CONV_2D', 4: 'DEPTHWISE_CONV_2D', 6: 'DEQUANTIZE',
       9: 'FULLY_CONNECTED', 14: 'LOGISTIC', 17: 'MAX_POOL_2D', 18: 'MUL', 19: 'RELU', 21: 'RELU6', 22: 'RESHAPE',
       23: 'RESIZE_BILINEAR', 25: 'SOFTMAX', 28: 'TANH', 32: 'CUSTOM', 34: 'PAD', 39: 'TRANSPOSE', 40: 'MEAN', 41: 'SUB',
       42: 'DIV', 45: 'STRIDED_SLICE', 54: 'PRELU', 60: 'PADV2', 65: 'SLICE', 67: 'TRANSPOSE_CONV', 74: 'SUM',
       83: 'PACK', 96: 'RESIZE_NEAREST_NEIGHBOR', 97: 'LEAKY_RELU', 116: 'HARD_SWISH'}
TYPES = {0: 'float32', 1: 'float16', 2: 'int32', 3: 'uint8', 4: 'int64', 9: 'int8'}
ACT = {0: None, 1: 'relu', 2: 'relu_n1_to_1', 3: 'relu6', 4: 'tanh'}
PAD = {0: 'same', 1: 'valid'}


class Table:
    def __init__(self, buf, pos):
        self.b, self.pos = buf, pos
        self._check(pos, 4)
        vt = pos - struct.unpack_from('<i', buf, pos)[0]
        self._check(vt, 4)
        self.vt = vt
        self.vlen = struct.unpack_from('<H', buf, vt)[0]

    def _check(self, p, n):
        if p < 0 or p + n > len(self.b):
            raise ValueError(f'offset {p}+{n} out of range')

    def _field(self, i):
        o = 4 + 2 * i
        if o + 2 > self.vlen:
            return 0
        return struct.unpack_from('<H', self.b, self.vt + o)[0]

    def scalar(self, i, fmt, default=0):
        f = self._field(i)
        if not f:
            return default
        self._check(self.pos + f, struct.calcsize(fmt))
        return struct.unpack_from('<' + fmt, self.b, self.pos + f)[0]

    def _ref(self, i):
        f = self._field(i)
        if not f:
            return None
        p = self.pos + f
        self._check(p, 4)
        return p + struct.unpack_from('<I', self.b, p)[0]

    def table(self, i):
        p = self._ref(i)
        return None if p is None else Table(self.b, p)

    def vector(self, i):
        """(start, length) of a vector field, or None."""
        p = self._ref(i)
        if p is None:
            return None
        self._check(p, 4)
        n = struct.unpack_from('<I', self.b, p)[0]
        return p + 4, n

    def scalars(self, i, fmt):
        v = self.vector(i)
        if v is None:
            return []
        s, n = v
        size = struct.calcsize(fmt)
        self._check(s, n * size)
        return list(struct.unpack_from('<%d%s' % (n, fmt), self.b, s))

    def bytes(self, i):
        v = self.vector(i)
        if v is None:
            return b''
        s, n = v
        self._check(s, n)
        return self.b[s:s + n]

    def string(self, i):
        return self.bytes(i).decode('utf-8')

    def tables(self, i):
        v = self.vector(i)
        if v is None:
            return []
        s, n = v
        out = []
        for k in range(n):
            p = s + 4 * k
            self._check(p, 4)
            out.append(Table(self.b, p + struct.unpack_from('<I', self.b, p)[0]))
        return out


def options(op_name, t):
    if t is None:
        return {}
    if op_name == 'CONV_2D':
        return dict(padding=PAD[t.scalar(0, 'b')], stride=(t.scalar(2, 'i', 1), t.scalar(1, 'i', 1)), act=ACT[t.scalar(3, 'b')],
                    dilation=(t.scalar(5, 'i', 1), t.scalar(4, 'i', 1)))
    if op_name == 'DEPTHWISE_CONV_2D':
        return dict(padding=PAD[t.scalar(0, 'b')], stride=(t.scalar(2, 'i', 1), t.scalar(1, 'i', 1)), mult=t.scalar(3, 'i', 1),
                    act=ACT[t.scalar(4, 'b')], dilation=(t.scalar(6, 'i', 1), t.scalar(5, 'i', 1)))
    if op_name in ('MAX_POOL_2D', 'AVERAGE_POOL_2D'):
        return dict(padding=PAD[t.scalar(0, 'b')], stride=(t.scalar(2, 'i', 1), t.scalar(1, 'i', 1)),
                    filter=(t.scalar(4, 'i', 1), t.scalar(3, 'i', 1)), act=ACT[t.scalar(5, 'b')])
    if op_name in ('ADD', 'MUL', 'SUB', 'DIV'):
        return dict(act=ACT[t.scalar(0, 'b')])
    if op_name == 'CONCATENATION':
        return dict(axis=t.scalar(0, 'i'), act=ACT[t.scalar(1, 'b')])
    if op_name == 'RESHAPE':
        return dict(new_shape=t.scalars(0, 'i'))
    if op_name == 'FULLY_CONNECTED':
        return dict(act=ACT[t.scalar(0, 'b')], keep_dims=bool(t.scalar(2, 'b')))
    if op_name in ('RESIZE_BILINEAR', 'RESIZE_NEAREST_NEIGHBOR'):
        return dict(align_corners=bool(t.scalar(2 if op_name == 'RESIZE_BILINEAR' else 0, 'b')), half_pixel=bool(t.scalar(3 if op_name == 'RESIZE_BILINEAR' else 1, 'b')))
    if op_name == 'LEAKY_RELU':
        return dict(alpha=t.scalar(0, 'f'))
    if op_name == 'MEAN':
        return dict(keep_dims=bool(t.scalar(0, 'b')))
    return {}


def read(path):
    buf = open(path, 'rb').read()
    model = Table(buf, struct.unpack_from('<I', buf, 0)[0])
    codes = []
    for oc in model.tables(1):
        code = max(oc.scalar(0, 'b'), oc.scalar(3, 'i'))
        codes.append(OPS.get(code, f'OP{code}') if code != 32 else 'CUSTOM:' + oc.string(1))
    buffers = []
    for b in model.tables(4):
        if b.scalar(1, 'Q') or b.scalar(2, 'Q'):
            off, size = b.scalar(1, 'Q'), b.scalar(2, 'Q')
            if off + size > len(buf):
                raise ValueError('external buffer out of range')
            buffers.append(buf[off:off + size])
        else:
            buffers.append(b.bytes(0))
    sg = model.tables(2)
    if len(sg) != 1:
        raise ValueError('expected one subgraph')
    sg = sg[0]
    tensors = []
    for t in sg.tables(0):
        q = t.table(4)
        tensors.append(dict(name=t.string(3), shape=t.scalars(0, 'i'), type=TYPES.get(t.scalar(1, 'b'), t.scalar(1, 'b')),
                            data=buffers[t.scalar(2, 'I')] if t.scalar(2, 'I') < len(buffers) else b'',
                            scale=q.scalars(2, 'f') if q else [], zero=q.scalars(3, 'q') if q else []))
    ops = []
    for o in sg.tables(3):
        name = codes[o.scalar(0, 'I')]
        ops.append(dict(op=name, inputs=o.scalars(1, 'i'), outputs=o.scalars(2, 'i'), opts=options(name, o.table(4)),
                        custom=bytes(o.bytes(5))))
    return dict(tensors=tensors, ops=ops, inputs=sg.scalars(1, 'i'), outputs=sg.scalars(2, 'i'))


if __name__ == '__main__':
    import sys, collections
    for p in sys.argv[1:]:
        m = read(p)
        print(p, len(m['ops']), 'ops', collections.Counter(o['op'] for o in m['ops']))
        print('  types', collections.Counter(t['type'] for t in m['tensors']))
        for i in m['inputs']:
            print('  in ', m['tensors'][i]['name'], m['tensors'][i]['shape'], m['tensors'][i]['type'])
        for i in m['outputs']:
            print('  out', m['tensors'][i]['name'], m['tensors'][i]['shape'], m['tensors'][i]['type'])
        print('  opts', collections.Counter(str(sorted((k, str(v)) for k, v in o['opts'].items() if k in ('padding', 'act', 'stride', 'dilation', 'mult', 'filter'))) for o in m['ops'] if o['op'] in ('CONV_2D', 'DEPTHWISE_CONV_2D', 'MAX_POOL_2D')).most_common(12))
