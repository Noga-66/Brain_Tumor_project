<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Brain Animation</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&display=swap');

  * { margin: 0; padding: 0; box-sizing: border-box; }

  body {
    background: #020408;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 100vh;
    overflow: hidden;
    font-family: 'Orbitron', monospace;
  }

  .bg-grid {
    position: fixed; inset: 0; pointer-events: none; z-index: 0;
    background-image:
      linear-gradient(rgba(0,220,255,0.03) 1px, transparent 1px),
      linear-gradient(90deg, rgba(0,220,255,0.03) 1px, transparent 1px);
    background-size: 50px 50px;
  }

  .container {
    position: relative; z-index: 1;
    display: flex; flex-direction: column;
    align-items: center; gap: 24px;
  }

  .title {
    font-size: clamp(1.2rem, 3vw, 2rem);
    font-weight: 900;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    background: linear-gradient(90deg, #00dcff, #a855f7, #ff4d8d);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: titleGlow 3s ease-in-out infinite alternate;
  }

  @keyframes titleGlow {
    from { filter: drop-shadow(0 0 8px rgba(0,220,255,0.4)); }
    to   { filter: drop-shadow(0 0 20px rgba(168,85,247,0.6)); }
  }

  canvas {
    display: block;
    filter: drop-shadow(0 0 30px rgba(0,220,255,0.3));
  }

  .stats {
    display: flex; gap: 32px;
  }
  .stat {
    text-align: center;
    animation: fadeIn 1s ease both;
  }
  .stat:nth-child(2) { animation-delay: 0.2s; }
  .stat:nth-child(3) { animation-delay: 0.4s; }

  .stat-val {
    font-size: 1.5rem; font-weight: 700;
    color: #00dcff;
    text-shadow: 0 0 12px rgba(0,220,255,0.6);
  }
  .stat-label {
    font-size: 0.55rem; letter-spacing: 0.2em;
    color: #4a5568; margin-top: 4px;
  }

  @keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
  }

  .pulse-ring {
    position: absolute;
    border-radius: 50%;
    border: 1px solid rgba(0,220,255,0.15);
    animation: ringPulse 4s ease-out infinite;
    pointer-events: none;
  }
  .pulse-ring:nth-child(2) { animation-delay: 1.3s; }
  .pulse-ring:nth-child(3) { animation-delay: 2.6s; }

  @keyframes ringPulse {
    0%   { width: 200px; height: 200px; opacity: 0.5; margin: -100px; }
    100% { width: 600px; height: 600px; opacity: 0;   margin: -300px; }
  }
</style>
</head>
<body>
<div class="bg-grid"></div>

<div class="container">
  <div class="title">Neural Brain Scan</div>

  <div style="position:relative; display:flex; align-items:center; justify-content:center;">
    <div class="pulse-ring" style="position:absolute;top:50%;left:50%;"></div>
    <div class="pulse-ring" style="position:absolute;top:50%;left:50%;"></div>
    <div class="pulse-ring" style="position:absolute;top:50%;left:50%;"></div>
    <canvas id="brain" width="500" height="480"></canvas>
  </div>

  <div class="stats">
    <div class="stat">
      <div class="stat-val" id="s-neurons">0</div>
      <div class="stat-label">NEURONS ACTIVE</div>
    </div>
    <div class="stat">
      <div class="stat-val" id="s-signals">0</div>
      <div class="stat-label">SIGNALS / SEC</div>
    </div>
    <div class="stat">
      <div class="stat-val" id="s-sync">0%</div>
      <div class="stat-label">SYNC RATE</div>
    </div>
  </div>
</div>

<script>
const canvas = document.getElementById('brain');
const ctx = canvas.getContext('2d');
const W = canvas.width, H = canvas.height;
const CX = W / 2, CY = H / 2 - 10;

// ── Brain shape outline points (hand-crafted cubic bezier path) ──────────────
function drawBrainPath(ctx) {
  ctx.beginPath();
  // Right hemisphere
  ctx.moveTo(CX + 10, CY - 155);
  ctx.bezierCurveTo(CX + 80, CY - 175, CX + 170, CY - 140, CX + 185, CY - 70);
  ctx.bezierCurveTo(CX + 205, CY,       CX + 180, CY + 80,  CX + 140, CY + 110);
  ctx.bezierCurveTo(CX + 100, CY + 140, CX + 50,  CY + 155, CX + 10,  CY + 150);
  // Center bottom notch
  ctx.bezierCurveTo(CX + 5,   CY + 155, CX,       CY + 160, CX - 5,   CY + 155);
  // Left hemisphere
  ctx.bezierCurveTo(CX - 50,  CY + 155, CX - 100, CY + 140, CX - 140, CY + 110);
  ctx.bezierCurveTo(CX - 180, CY + 80,  CX - 205, CY,       CX - 185, CY - 70);
  ctx.bezierCurveTo(CX - 170, CY - 140, CX - 80,  CY - 175, CX - 10,  CY - 155);
  ctx.closePath();
}

// ── Sulci (brain folds) ──────────────────────────────────────────────────────
const sulci = [
  // right side folds
  { pts: [[CX+20,CY-140],[CX+60,CY-120],[CX+90,CY-90],[CX+100,CY-50]] },
  { pts: [[CX+60,CY-150],[CX+100,CY-130],[CX+140,CY-90],[CX+160,CY-40]] },
  { pts: [[CX+110,CY-130],[CX+150,CY-80],[CX+170,CY-20],[CX+155,CY+40]] },
  { pts: [[CX+30,CY-110],[CX+70,CY-60],[CX+90,CY-10],[CX+80,CY+50]] },
  { pts: [[CX+70,CY-30],[CX+110,CY+10],[CX+130,CY+60],[CX+110,CY+100]] },
  { pts: [[CX+20,CY+60],[CX+60,CY+90],[CX+80,CY+120],[CX+60,CY+145]] },
  // left side folds
  { pts: [[CX-20,CY-140],[CX-60,CY-120],[CX-90,CY-90],[CX-100,CY-50]] },
  { pts: [[CX-60,CY-150],[CX-100,CY-130],[CX-140,CY-90],[CX-160,CY-40]] },
  { pts: [[CX-110,CY-130],[CX-150,CY-80],[CX-170,CY-20],[CX-155,CY+40]] },
  { pts: [[CX-30,CY-110],[CX-70,CY-60],[CX-90,CY-10],[CX-80,CY+50]] },
  { pts: [[CX-70,CY-30],[CX-110,CY+10],[CX-130,CY+60],[CX-110,CY+100]] },
  { pts: [[CX-20,CY+60],[CX-60,CY+90],[CX-80,CY+120],[CX-60,CY+145]] },
  // top center
  { pts: [[CX-10,CY-155],[CX-5,CY-100],[CX,CY-40],[CX+5,CY+20]] },
];

// ── Neural nodes ─────────────────────────────────────────────────────────────
const NODES = 55;
const nodes = [];
for (let i = 0; i < NODES; i++) {
  const angle = Math.random() * Math.PI * 2;
  const r = 60 + Math.random() * 110;
  const side = Math.random() > 0.5 ? 1 : -1;
  nodes.push({
    x: CX + side * (20 + Math.abs(Math.cos(angle)) * r * 0.9),
    y: CY + Math.sin(angle) * r * 0.75,
    r: 2 + Math.random() * 3,
    phase: Math.random() * Math.PI * 2,
    speed: 0.02 + Math.random() * 0.03,
    color: Math.random() > 0.6 ? '#00dcff' : Math.random() > 0.5 ? '#a855f7' : '#ff4d8d',
    connections: []
  });
}

// Connect nearby nodes
nodes.forEach((n, i) => {
  nodes.forEach((m, j) => {
    if (i >= j) return;
    const d = Math.hypot(n.x - m.x, n.y - m.y);
    if (d < 90 && n.connections.length < 4 && m.connections.length < 4) {
      n.connections.push(j);
    }
  });
});

// ── Signals traveling along connections ──────────────────────────────────────
const signals = [];
function spawnSignal() {
  const n = nodes[Math.floor(Math.random() * nodes.length)];
  if (!n.connections.length) return;
  const to = n.connections[Math.floor(Math.random() * n.connections.length)];
  signals.push({
    from: nodes.indexOf(n),
    to,
    t: 0,
    speed: 0.012 + Math.random() * 0.018,
    color: n.color,
    size: 3 + Math.random() * 3
  });
}

// ── Scan line ─────────────────────────────────────────────────────────────────
let scanY = CY - 160;
let scanDir = 1;

// ── Stats counters ────────────────────────────────────────────────────────────
let frame = 0;
let activeCount = 0;
let signalCount = 0;

// ── Main render loop ──────────────────────────────────────────────────────────
function render() {
  ctx.clearRect(0, 0, W, H);
  frame++;

  // Spawn signals
  if (frame % 8 === 0) spawnSignal();

  // ── Brain base fill ──
  drawBrainPath(ctx);
  const grad = ctx.createRadialGradient(CX, CY - 20, 30, CX, CY, 200);
  grad.addColorStop(0,   'rgba(0,40,80,0.9)');
  grad.addColorStop(0.5, 'rgba(0,20,50,0.95)');
  grad.addColorStop(1,   'rgba(0,5,20,0.98)');
  ctx.fillStyle = grad;
  ctx.fill();

  // ── Clip all drawing inside brain ──
  ctx.save();
  drawBrainPath(ctx);
  ctx.clip();

  // ── Inner glow ──
  drawBrainPath(ctx);
  const innerGlow = ctx.createRadialGradient(CX, CY, 0, CX, CY, 200);
  innerGlow.addColorStop(0,   'rgba(0,180,255,0.08)');
  innerGlow.addColorStop(0.6, 'rgba(100,0,255,0.04)');
  innerGlow.addColorStop(1,   'transparent');
  ctx.fillStyle = innerGlow;
  ctx.fill();

  // ── Connection lines ──
  nodes.forEach((n, i) => {
    n.connections.forEach(j => {
      const m = nodes[j];
      ctx.beginPath();
      ctx.moveTo(n.x, n.y);
      ctx.lineTo(m.x, m.y);
      ctx.strokeStyle = 'rgba(0,150,255,0.08)';
      ctx.lineWidth = 0.5;
      ctx.stroke();
    });
  });

  // ── Sulci ──
  sulci.forEach(s => {
    const pts = s.pts;
    ctx.beginPath();
    ctx.moveTo(pts[0][0], pts[0][1]);
    for (let i = 1; i < pts.length - 1; i++) {
      const mx = (pts[i][0] + pts[i+1][0]) / 2;
      const my = (pts[i][1] + pts[i+1][1]) / 2;
      ctx.quadraticCurveTo(pts[i][0], pts[i][1], mx, my);
    }
    ctx.strokeStyle = 'rgba(0,80,150,0.35)';
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // fold highlight
    ctx.beginPath();
    ctx.moveTo(pts[0][0]+1, pts[0][1]+1);
    for (let i = 1; i < pts.length - 1; i++) {
      const mx = (pts[i][0] + pts[i+1][0]) / 2 + 1;
      const my = (pts[i][1] + pts[i+1][1]) / 2 + 1;
      ctx.quadraticCurveTo(pts[i][0]+1, pts[i][1]+1, mx, my);
    }
    ctx.strokeStyle = 'rgba(0,200,255,0.1)';
    ctx.lineWidth = 0.5;
    ctx.stroke();
  });

  // ── Traveling signals ──
  for (let i = signals.length - 1; i >= 0; i--) {
    const s = signals[i];
    s.t += s.speed;
    if (s.t >= 1) { signals.splice(i, 1); signalCount++; continue; }

    const a = nodes[s.from], b = nodes[s.to];
    const x = a.x + (b.x - a.x) * s.t;
    const y = a.y + (b.y - a.y) * s.t;

    // trail
    const trailLen = 0.15;
    const t0 = Math.max(0, s.t - trailLen);
    const tx = a.x + (b.x - a.x) * t0;
    const ty = a.y + (b.y - a.y) * t0;
    const trailGrad = ctx.createLinearGradient(tx, ty, x, y);
    trailGrad.addColorStop(0, 'transparent');
    trailGrad.addColorStop(1, s.color + 'cc');
    ctx.beginPath();
    ctx.moveTo(tx, ty);
    ctx.lineTo(x, y);
    ctx.strokeStyle = trailGrad;
    ctx.lineWidth = s.size * 0.6;
    ctx.stroke();

    // head glow
    ctx.beginPath();
    ctx.arc(x, y, s.size, 0, Math.PI * 2);
    ctx.fillStyle = s.color;
    ctx.shadowColor = s.color;
    ctx.shadowBlur = 12;
    ctx.fill();
    ctx.shadowBlur = 0;
  }

  // ── Neural nodes ──
  activeCount = 0;
  nodes.forEach(n => {
    const pulse = Math.sin(frame * n.speed + n.phase);
    const bright = 0.5 + pulse * 0.5;
    if (bright > 0.7) activeCount++;
    ctx.beginPath();
    ctx.arc(n.x, n.y, n.r * bright, 0, Math.PI * 2);
    ctx.fillStyle = n.color;
    ctx.globalAlpha = 0.4 + bright * 0.6;
    ctx.shadowColor = n.color;
    ctx.shadowBlur = 8 + pulse * 8;
    ctx.fill();
    ctx.globalAlpha = 1;
    ctx.shadowBlur = 0;
  });

  // ── Scan line ──
  scanY += scanDir * 1.2;
  if (scanY > CY + 160) scanDir = -1;
  if (scanY < CY - 160) scanDir = 1;

  const scanGrad = ctx.createLinearGradient(CX - 200, scanY, CX + 200, scanY);
  scanGrad.addColorStop(0,   'transparent');
  scanGrad.addColorStop(0.3, 'rgba(0,220,255,0.06)');
  scanGrad.addColorStop(0.5, 'rgba(0,220,255,0.18)');
  scanGrad.addColorStop(0.7, 'rgba(0,220,255,0.06)');
  scanGrad.addColorStop(1,   'transparent');
  ctx.fillStyle = scanGrad;
  ctx.fillRect(CX - 200, scanY - 3, 400, 6);

  ctx.restore(); // end clip

  // ── Brain outline glow ──
  drawBrainPath(ctx);
  ctx.strokeStyle = 'rgba(0,180,255,0.5)';
  ctx.lineWidth = 1.5;
  ctx.shadowColor = '#00dcff';
  ctx.shadowBlur = 20;
  ctx.stroke();
  ctx.shadowBlur = 0;

  // secondary outer glow
  drawBrainPath(ctx);
  ctx.strokeStyle = 'rgba(168,85,247,0.2)';
  ctx.lineWidth = 4;
  ctx.shadowColor = '#a855f7';
  ctx.shadowBlur = 30;
  ctx.stroke();
  ctx.shadowBlur = 0;

  // ── Center dividing line ──
  ctx.save();
  drawBrainPath(ctx);
  ctx.clip();
  ctx.beginPath();
  ctx.moveTo(CX, CY - 150);
  ctx.bezierCurveTo(CX - 10, CY - 50, CX + 10, CY + 50, CX, CY + 150);
  ctx.strokeStyle = 'rgba(0,180,255,0.2)';
  ctx.lineWidth = 1;
  ctx.stroke();
  ctx.restore();

  // ── Update stats ──
  if (frame % 20 === 0) {
    document.getElementById('s-neurons').textContent = activeCount;
    document.getElementById('s-signals').textContent = Math.floor(signalCount * 2);
    const sync = Math.floor(60 + Math.sin(frame * 0.01) * 30);
    document.getElementById('s-sync').textContent = sync + '%';
    signalCount = 0;
  }

  requestAnimationFrame(render);
}

render();
</script>
</body>
</html>
