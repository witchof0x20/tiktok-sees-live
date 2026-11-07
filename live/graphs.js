// Rolling time-series for the largest face: attractiveness, age, P(male) as single-series lines (one y-scale each,
// small multiples rather than shared axes) and the 7-way mood vector as a stacked area. Canvas, redrawn per frame.
const MOOD = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'];
const WINDOW = 15000;

const css = (el, name) => getComputedStyle(el).getPropertyValue(name).trim();

function makeChart(root, { title, unit, min, max, ticks, fmt, tick = fmt, stacked }) {
  const box = document.createElement('figure');
  box.className = 'chart' + (stacked ? ' wide' : '');
  box.innerHTML = `<figcaption><span class="t"></span><span class="v"></span></figcaption><div class="plot"><canvas></canvas><div class="tip" hidden></div></div>${stacked ? '<ul class="legend"></ul>' : ''}`;
  box.querySelector('.t').textContent = title;
  root.append(box);
  const c = { box, canvas: box.querySelector('canvas'), value: box.querySelector('.v'), tip: box.querySelector('.tip'), title, unit, min, max, ticks, fmt, tick, stacked, hover: null };
  if (stacked) {
    c.legend = MOOD.map((name, i) => {
      const li = document.createElement('li');
      li.innerHTML = `<i style="background: var(--series-${i + 1})"></i><span></span><b>–</b>`;
      li.querySelector('span').textContent = name;
      box.querySelector('.legend').append(li);
      return li.querySelector('b');
    });
  }
  const plot = box.querySelector('.plot');
  plot.addEventListener('pointermove', e => { const r = c.canvas.getBoundingClientRect(); c.hover = (e.clientX - r.left) / r.width; });
  plot.addEventListener('pointerleave', () => { c.hover = null; c.tip.hidden = true; });
  return c;
}

export class Graphs {
  constructor(root) {
    this.root = root;
    this.samples = [];
    this.charts = {
      attractive: makeChart(root, { title: 'Attractiveness', min: 0, max: 100, ticks: [0, 50, 100], fmt: v => v.toFixed(1), tick: v => `${v}` }),
      age: makeChart(root, { title: 'Age guess', min: 0, max: 80, ticks: [0, 20, 40, 60, 80], fmt: v => v.toFixed(1), tick: v => `${v}` }),
      boy: makeChart(root, { title: 'P(male)', min: 0, max: 1, ticks: [0, 0.5, 1], fmt: v => `${(v * 100).toFixed(0)}%` }),
      exp: makeChart(root, { title: 'Mood', min: 0, max: 1, ticks: [0, 0.5, 1], fmt: v => `${(v * 100).toFixed(0)}%`, stacked: true }),
    };
    const loop = () => { this.render(); requestAnimationFrame(loop); };
    requestAnimationFrame(loop);
  }

  // s = null when no face is visible, so lines break instead of bridging the gap.
  push(t, s) {
    this.samples.push({ t, s });
    while (this.samples.length && t - this.samples[0].t > WINDOW + 1000) this.samples.shift();
  }

  render() {
    if (this.root.hidden || !this.samples.length) return;
    const now = performance.now();
    for (const [key, c] of Object.entries(this.charts)) this.draw(key, c, now);
  }

  draw(key, c, now) {
    const dpr = devicePixelRatio || 1, cv = c.canvas, W = cv.clientWidth, H = cv.clientHeight;
    if (!W || !H) return;
    if (cv.width !== Math.round(W * dpr) || cv.height !== Math.round(H * dpr)) { cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr); }
    const g = cv.getContext('2d');
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, W, H);
    const L = 40, R = 6, T = 7, B = 18, pw = W - L - R, ph = H - T - B;
    const X = t => L + pw * (1 - (now - t) / WINDOW), Y = v => T + ph * (1 - (v - c.min) / (c.max - c.min));
    const ink2 = css(this.root, '--text-secondary'), ink3 = css(this.root, '--text-muted'), grid = css(this.root, '--grid'), surface = css(this.root, '--surface-1');

    // Recessive grid and axis labels.
    g.font = '11px ui-sans-serif, system-ui, sans-serif'; g.textBaseline = 'middle'; g.textAlign = 'right';
    g.lineWidth = 1; g.strokeStyle = grid;
    for (const v of c.ticks) {
      const y = Math.round(Y(v)) + 0.5;
      g.beginPath(); g.moveTo(L, y); g.lineTo(L + pw, y); g.stroke();
      g.fillStyle = ink3; g.fillText(c.tick(v), L - 6, y);
    }
    g.textAlign = 'left'; g.textBaseline = 'alphabetic'; g.fillText('−15 s', L, H - 4);
    g.textAlign = 'right'; g.fillText('now', L + pw, H - 4);

    const pts = this.samples.filter(p => now - p.t <= WINDOW);
    // Split into runs of consecutive samples that have a value.
    const runs = [];
    let run = [];
    for (const p of pts) {
      const v = p.s && (c.stacked ? p.s.exp : p.s[key]);
      if (v == null || (key === 'attractive' && p.s.age < 16)) { if (run.length) runs.push(run); run = []; }
      else run.push({ t: p.t, v });
    }
    if (run.length) runs.push(run);

    g.save(); g.beginPath(); g.rect(L, T - 1, pw, ph + 2); g.clip();
    if (c.stacked) {
      for (const r of runs) {
        let lower = r.map(() => 0);
        MOOD.forEach((_, k) => {
          const upper = r.map((p, i) => lower[i] + p.v[k]);
          g.beginPath();
          r.forEach((p, i) => (i ? g.lineTo : g.moveTo).call(g, X(p.t), Y(upper[i])));
          for (let i = r.length - 1; i >= 0; i--) g.lineTo(X(r[i].t), Y(lower[i]));
          g.closePath();
          g.fillStyle = css(this.root, `--series-${k + 1}`); g.fill();
          // 2px surface gap between layers.
          g.beginPath();
          r.forEach((p, i) => (i ? g.lineTo : g.moveTo).call(g, X(p.t), Y(upper[i])));
          g.strokeStyle = surface; g.lineWidth = 2; g.stroke();
          lower = upper;
        });
      }
    } else {
      g.strokeStyle = css(this.root, '--series-1'); g.lineWidth = 2; g.lineJoin = 'round'; g.lineCap = 'round';
      for (const r of runs) {
        g.beginPath();
        r.forEach((p, i) => (i ? g.lineTo : g.moveTo).call(g, X(p.t), Y(p.v)));
        g.stroke();
      }
    }
    g.restore();

    // Live readout (text ink, never series color).
    const last = runs.length && runs[runs.length - 1][runs[runs.length - 1].length - 1];
    const fresh = last && now - last.t < 500;
    if (c.stacked) {
      c.legend.forEach((b, k) => { b.textContent = fresh ? `${Math.round(last.v[k] * 100)}%` : '–'; });
      const top = fresh ? last.v.map((v, k) => [MOOD[k], v]).sort((a, b) => b[1] - a[1])[0] : null;
      c.value.textContent = top ? `${top[0]} ${Math.round(top[1] * 100)}%` : '–';
    } else c.value.textContent = fresh ? c.fmt(last.v) : '–';

    // Hover: crosshair + tooltip for the nearest sample.
    if (c.hover != null) {
      const th = now - WINDOW * (1 - (c.hover * W - L) / pw);
      let best = null;
      for (const r of runs) for (const p of r) if (!best || Math.abs(p.t - th) < Math.abs(best.t - th)) best = p;
      if (best) {
        const x = Math.round(X(best.t)) + 0.5;
        g.strokeStyle = ink2; g.lineWidth = 1; g.beginPath(); g.moveTo(x, T); g.lineTo(x, T + ph); g.stroke();
        if (!c.stacked) {
          g.beginPath(); g.arc(X(best.t), Y(best.v), 4, 0, Math.PI * 2);
          g.fillStyle = css(this.root, '--series-1'); g.fill(); g.lineWidth = 2; g.strokeStyle = surface; g.stroke();
        }
        const ago = ((now - best.t) / 1000).toFixed(1);
        c.tip.textContent = '';
        const head = document.createElement('div'); head.className = 'h'; head.textContent = `${ago} s ago`; c.tip.append(head);
        const rows = c.stacked ? MOOD.map((n, k) => [n, `${Math.round(best.v[k] * 100)}%`, k]) : [[c.title, c.fmt(best.v), 0]];
        for (const [n, v, k] of rows) {
          const row = document.createElement('div');
          row.innerHTML = '<i></i><span></span><b></b>';
          row.querySelector('i').style.background = `var(--series-${k + 1})`;
          row.querySelector('span').textContent = n; row.querySelector('b').textContent = v;
          c.tip.append(row);
        }
        c.tip.hidden = false;
        const left = Math.min(Math.max(x + 10, 0), W - c.tip.offsetWidth - 4);
        c.tip.style.left = `${x + 10 + c.tip.offsetWidth > W ? x - c.tip.offsetWidth - 10 : left}px`;
      }
    }
  }
}
