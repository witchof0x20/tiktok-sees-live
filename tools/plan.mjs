// Prints the fusion plan for both nets.
import fs from 'node:fs';
import { parseNet } from '../live/gpu/graph.js';
import { planNet, DEFAULTS } from '../live/gpu/fuse.js';
const dir = process.argv[2], WANT = { expnet: ['predict_attractive', 'prob_exp', 'predict_valence', 'predict_arousal'], agenet: ['predict_age', 'prob_gender'] };
let tot = 0, totBase = 0; const levels = new Map();
for (const name of ['expnet', 'agenet']) {
  const net = parseNet(JSON.parse(fs.readFileSync(`${dir}/${name}.json`)), WANT[name]);
  const groups = planNet(net, { ...DEFAULTS, ...JSON.parse(process.argv[3] || '{}') });
  console.log(`${name}: ${net.order.length - 1} ops -> ${groups.length} kernels`);
  for (const g of groups) {
    const p = g.plan; tot += p.work; totBase += p.base;
    levels.set(g.level, (levels.get(g.level) || 0) + 1);
    console.log(` L${String(g.level).padStart(2)} ${String(g.members.size).padStart(2)} ops T=${String(p.T).padStart(2)} S=${String(p.S).padStart(2)} wg=${String(p.nwg).padStart(4)} smem=${String(p.bytes).padStart(5)} redo=${(p.work / p.base).toFixed(2)} P${p.P}Q${p.Q} shared=[${p.stages.map(s => `${s.n.op}:${s.P}x${s.Q}`).join(',')}] -> ${g.F.name}${g.F.toGap ? ' (gap)' : ''}`);
  }
}
console.log('total work ratio', (tot / totBase).toFixed(2), 'dispatches with horizontal fusion', levels.size);
