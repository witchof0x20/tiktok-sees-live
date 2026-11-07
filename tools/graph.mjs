// Dumps each op with output shape and consumer list.
import fs from 'node:fs';
const [dir,name]=process.argv.slice(2);
const spec=JSON.parse(fs.readFileSync(`${dir}/${name}.json`));
const shape={},cons={};
const ins=o=>o.op==='data'?[]:o.op==='add'?[o.a,o.b]:o.op==='concat'?o.tops:[o.top];
for(const o of spec.ops)for(const t of ins(o))(cons[t]??=[]).push(o.bot);
for(const o of spec.ops){
  const out=o.bot??o.name;
  if(o.op==='data')shape[out]=[o.c,o.h,o.w];
  else if(o.op==='conv'||o.op==='dw'){const [c,h,w]=shape[o.top];shape[out]=[o.o,Math.floor((h+2*o.p-o.kh)/o.s)+1,Math.floor((w+2*o.p-o.kw)/o.s)+1];}
  else if(o.op==='add')shape[out]=shape[o.a];
  else if(o.op==='concat'){const f=shape[o.tops[0]];shape[out]=[o.tops.reduce((a,t)=>a+shape[t][0],0),f[1],f[2]];}
  else if(o.op==='gap')shape[out]=[shape[o.top][0]];
  else if(o.op==='fc')shape[out]=[o.o];
  else shape[out]=shape[o.top];
  const k=o.kh?`${o.kh}x${o.kw}s${o.s}p${o.p}${o.act?'R':''}`:(o.act?'R':'');
  console.log(o.op.padEnd(7),k.padEnd(9),String(shape[out]).padEnd(11),out.padEnd(44),'<-',ins(o).join(','),'  ->',(cons[out]||[]).join(','));
}
