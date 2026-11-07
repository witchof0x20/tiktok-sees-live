// Interval analysis: worst-case conv accumulator magnitude per layer, to decide where i32 accumulation is exact.
import fs from 'node:fs';
const dir=process.argv[2];
const rs=(x,s)=>s<=0?x*Math.pow(2,-s):Math.floor(x/Math.pow(2,s)+.5);
const sat=x=>Math.max(-32768,Math.min(32767,x));
for(const name of ['expnet','agenet']){
  const spec=JSON.parse(fs.readFileSync(`${dir}/${name}.json`)),buf=fs.readFileSync(`${dir}/${name}.bin`);
  const ab=buf.buffer.slice(buf.byteOffset,buf.byteOffset+buf.length);
  const R={};let over=0,minW=0,maxAcc=0,shifts=new Set(),bshifts=new Set(),addsh=new Set();
  for(const o of spec.ops){
    if(o.op==='data'){R[o.name]=Array.from({length:o.c},()=>[-128,127]);}
    else if(o.op==='conv'||o.op==='dw'){
      const X=R[o.top],dw=o.op==='dw',K=o.kh*o.kw;
      const wt=new Int16Array(ab,o.wOff,dw?K*o.o:o.o*K*o.i),bt=o.bias?new Int32Array(ab,o.bOff,o.o):null;
      const out=[];let lay=0;shifts.add(`${o.op}:${X.f??''}`);
      for(let c=0;c<o.o;c++){
        const b=bt?rs(bt[c],o.bf-(spec.ops.find(q=>q.bot===o.top||q.name===o.top).of??spec.ops[0].frac)-o.wf):0;
        let lo=b,hi=b;
        for(let t=0;t<K;t++)for(let i=0;i<(dw?1:o.i);i++){
          const w=dw?wt[t*o.o+c]:wt[(c*K+t)*o.i+i];minW=Math.min(minW,w);const [a,z]=X[dw?c:i];
          lo+=Math.min(w*a,w*z);hi+=Math.max(w*a,w*z);
        }
        lay=Math.max(lay,Math.abs(lo),Math.abs(hi));
        const xf=(spec.ops.find(q=>q.bot===o.top||q.name===o.top)),inf=xf.op==='data'?xf.frac:xf.of,sh=inf+o.wf-o.of;shifts.add(sh);bshifts.add(o.bf-(inf+o.wf));
        let l2=sat(rs(lo,sh)),h2=sat(rs(hi,sh));if(o.act){l2=Math.max(0,l2);h2=Math.max(0,h2);}out.push([l2,h2]);
      }
      if(lay>=2**31){over++;console.log('  OVER',name,o.bot,Math.log2(lay).toFixed(2));}
      maxAcc=Math.max(maxAcc,lay);R[o.bot]=out;
    } else if(o.op==='add'){
      const A=R[o.a],B=R[o.b],fa=spec.ops.find(q=>q.bot===o.a).of,fb=spec.ops.find(q=>q.bot===o.b).of;addsh.add(`${fa-o.of},${fb-o.of}`);
      R[o.bot]=A.map((a,c)=>{let l=sat(rs(a[0],fa-o.of)+rs(B[c][0],fb-o.of)),h=sat(rs(a[1],fa-o.of)+rs(B[c][1],fb-o.of));if(o.act){l=Math.max(0,l);h=Math.max(0,h);}return [l,h];});
    } else if(o.op==='concat'){R[o.bot]=o.tops.flatMap(t=>{const f=spec.ops.find(q=>q.bot===t).of;addsh.add('cat'+(f-o.of));return R[t].map(([l,h])=>[sat(rs(l,f-o.of)),sat(rs(h,f-o.of))]);});}
    else R[o.bot]=R[o.top];
  }
  console.log(name,'layers over 2^31:',over,'max log2',Math.log2(maxAcc).toFixed(2),'min weight',minW,'conv shifts',[...shifts].filter(x=>typeof x==='number').sort((a,b)=>a-b).join(','),'bias shifts',[...bshifts].sort((a,b)=>a-b).join(','),'add shifts',[...addsh].join(' '));
}
