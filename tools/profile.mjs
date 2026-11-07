// Prints per-net op counts, MACs, and worst-case conv accumulator magnitudes.
import fs from 'node:fs';
const dir=process.argv[2];
for(const name of ['expnet','agenet']){
  const spec=JSON.parse(fs.readFileSync(`${dir}/${name}.json`)),buf=fs.readFileSync(`${dir}/${name}.bin`);
  const ab=buf.buffer.slice(buf.byteOffset,buf.byteOffset+buf.length);
  const shape={},kinds={};let macs=0,worst=0,worstOp='';
  for(const o of spec.ops){
    kinds[o.op]=(kinds[o.op]||0)+1;
    if(o.op==='data')shape[o.name]=[o.c,o.h,o.w];
    else if(o.op==='conv'||o.op==='dw'){
      const [c,h,w]=shape[o.top],oh=Math.floor((h+2*o.p-o.kh)/o.s)+1,ow=Math.floor((w+2*o.p-o.kw)/o.s)+1,dw=o.op==='dw';
      macs+=oh*ow*o.o*o.kh*o.kw*(dw?1:o.i);shape[o.bot]=[o.o,oh,ow];
      const wt=new Int16Array(ab,o.wOff,dw?o.kh*o.kw*o.o:o.o*o.kh*o.kw*o.i);
      for(let k=0;k<o.o;k++){let s=0;if(dw){for(let t=0;t<o.kh*o.kw;t++)s+=Math.abs(wt[t*o.o+k]);}else{for(let t=0;t<o.kh*o.kw*o.i;t++)s+=Math.abs(wt[k*o.kh*o.kw*o.i+t]);}
        if(s*32768>worst){worst=s*32768;worstOp=o.bot;}}
    } else if(o.op==='add')shape[o.bot]=shape[o.a];
    else if(o.op==='concat'){const f=shape[o.tops[0]];shape[o.bot]=[o.tops.reduce((a,t)=>a+shape[t][0],0),f[1],f[2]];}
    else if(o.op==='gap')shape[o.bot]=[shape[o.top][0],1,1];
    else shape[o.bot]=shape[o.top];
  }
  console.log(name,kinds,'MMACs',(macs/1e6).toFixed(1),'worst |acc| bound',worst.toExponential(2),'log2',Math.log2(worst).toFixed(1),worstOp);
  const fcs=spec.ops.filter(o=>o.op==='fc').map(o=>`${o.bot}:${o.i}->${o.o}`);console.log(' fc',fcs.join(' '));
  console.log(' sh<=0 count',spec.ops.filter(o=>(o.op==='conv'||o.op==='dw')).map(o=>o).length);
}
