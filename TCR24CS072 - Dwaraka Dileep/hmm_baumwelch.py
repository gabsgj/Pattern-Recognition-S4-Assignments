import numpy as np
import webbrowser
import os
import json
import tempfile

# ─────────────────────────────────────────
#  HMM STRUCTURE
# ─────────────────────────────────────────

states       = ["Rainy", "Sunny"]
observations = ["Walk", "Shop", "Clean"]
obs_seq      = [0, 1, 2, 0]

N = len(states)
M = len(observations)
T = len(obs_seq)

pi = np.array([0.6, 0.4])
A  = np.array([[0.7, 0.3],
               [0.4, 0.6]])
B  = np.array([[0.1, 0.4, 0.5],
               [0.6, 0.3, 0.1]])


# ─────────────────────────────────────────
#  FORWARD ALGORITHM
# ─────────────────────────────────────────

def forward(pi, A, B):
    alpha = np.zeros((T, N))
    for i in range(N):
        alpha[0][i] = pi[i] * B[i][obs_seq[0]]
    for t in range(1, T):
        for j in range(N):
            alpha[t][j] = np.sum(alpha[t-1] * A[:, j]) * B[j][obs_seq[t]]
    return alpha


# ─────────────────────────────────────────
#  BACKWARD ALGORITHM
# ─────────────────────────────────────────

def backward(A, B):
    beta = np.zeros((T, N))
    beta[T-1] = 1
    for t in reversed(range(T - 1)):
        for i in range(N):
            beta[t][i] = np.sum(A[i] * B[:, obs_seq[t+1]] * beta[t+1])
    return beta


# ─────────────────────────────────────────
#  BAUM-WELCH ALGORITHM
# ─────────────────────────────────────────

def baum_welch(iterations=15):
    global A, B, pi
    history = []

    for iteration in range(iterations):
        alpha = forward(pi, A, B)
        beta  = backward(A, B)

        xi    = np.zeros((T-1, N, N))
        gamma = np.zeros((T, N))

        for t in range(T - 1):
            denom = np.sum(alpha[t] * beta[t])
            for i in range(N):
                gamma[t][i] = (alpha[t][i] * beta[t][i]) / denom
                for j in range(N):
                    xi[t][i][j] = (
                        alpha[t][i] * A[i][j] *
                        B[j][obs_seq[t+1]] * beta[t+1][j]
                    ) / denom

        denom_last = np.sum(alpha[T-1] * beta[T-1])
        gamma[T-1] = (alpha[T-1] * beta[T-1]) / denom_last

        pi = gamma[0]

        for i in range(N):
            for j in range(N):
                A[i][j] = np.sum(xi[:, i, j]) / np.sum(gamma[:-1, i])

        for i in range(N):
            for k in range(M):
                num = sum(gamma[t][i] for t in range(T) if obs_seq[t] == k)
                den = np.sum(gamma[:, i])
                B[i][k] = num / den

        ll = float(np.log(np.sum(alpha[T-1])))

        history.append({
            "iteration": iteration + 1,
            "pi":    pi.tolist(),
            "A":     A.tolist(),
            "B":     B.tolist(),
            "alpha": alpha.tolist(),
            "beta":  beta.tolist(),
            "ll":    ll
        })

        print(f"  Iteration {iteration+1:>2}  |  log-likelihood: {ll:.6f}")

    return history


# ─────────────────────────────────────────
#  PRINT RESULTS
# ─────────────────────────────────────────

print("=" * 52)
print("   Hidden Markov Model - Baum-Welch Training")
print("=" * 52)
print(f"\n  States       : {states}")
print(f"  Observations : {observations}")
print(f"  Obs Sequence : {[observations[o] for o in obs_seq]}")
print("\n  Initial pi:", pi)
print("  Initial A:\n", A)
print("  Initial B:\n", B)

ITERS = 15
print(f"\n  Running Baum-Welch ({ITERS} iterations)...\n")
history = baum_welch(iterations=ITERS)

print(f"\n  Updated A:\n", np.round(A, 4))
print("  Updated B:\n", np.round(B, 4))
print("\n" + "=" * 52)
print("  Training complete. Opening visualizer...")
print("=" * 52 + "\n")


# ─────────────────────────────────────────
#  BUILD & OPEN HTML VISUALIZER
# ─────────────────────────────────────────

HISTORY_JSON = json.dumps(history)
ITERS_VAL    = ITERS

html_content = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>HMM Baum-Welch Visualizer</title>
<link href="https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;600;700&family=JetBrains+Mono:wght@300;400;500;700&display=swap" rel="stylesheet"/>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#080b10;--surface:#0e1319;--card:#131920;
  --border:#1e2a36;--border2:#263040;
  --rainy:#38bdf8;--sunny:#fbbf24;--green:#4ade80;
  --white:#f0f6ff;--muted:#4a6070;
}
html{font-size:15px}
body{background:var(--bg);color:var(--white);font-family:'JetBrains Mono',monospace;min-height:100vh}

header{
  background:var(--surface);border-bottom:2px solid var(--border2);
  padding:0 36px;height:64px;display:flex;align-items:center;gap:20px;
  position:sticky;top:0;z-index:200;
}
header h1{font-family:'Rajdhani',sans-serif;font-size:1.5rem;font-weight:700;letter-spacing:1px}
header h1 span{color:var(--rainy)}
.sep{width:1px;height:28px;background:var(--border2)}
.tag{font-size:0.6rem;letter-spacing:2px;text-transform:uppercase;color:var(--muted);padding:3px 10px;border:1px solid var(--border2);border-radius:2px}
.spacer{flex:1}
.live-ll{font-size:0.7rem;color:var(--green);letter-spacing:1px}

.page{padding:28px 36px 80px;max-width:1200px;margin:0 auto}
.sec{font-family:'Rajdhani',sans-serif;font-size:0.65rem;letter-spacing:3px;text-transform:uppercase;color:var(--muted);margin-bottom:14px;display:flex;align-items:center;gap:10px}
.sec::after{content:'';flex:1;height:1px;background:var(--border)}
.row{display:grid;gap:20px;margin-bottom:28px}
.col2{grid-template-columns:1fr 1fr}
.col3{grid-template-columns:1fr 1fr 1fr}
@media(max-width:700px){.col2,.col3{grid-template-columns:1fr}.page{padding:20px 16px 60px}header{padding:0 16px}}

.card{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:22px 24px}
.clabel{font-size:0.6rem;letter-spacing:2px;text-transform:uppercase;color:var(--muted);margin-bottom:14px;border-bottom:1px solid var(--border);padding-bottom:10px}

.ctrl{background:var(--card);border:1px solid var(--border2);border-radius:8px;padding:24px 28px;margin-bottom:28px;display:flex;align-items:center;gap:32px;flex-wrap:wrap}
.iter-big{font-family:'Rajdhani',sans-serif;font-size:3.5rem;font-weight:700;color:var(--rainy);line-height:1;min-width:60px}
.iter-lbl{font-size:0.6rem;letter-spacing:2px;color:var(--muted);text-transform:uppercase;margin-top:4px}
.slider-wrap{flex:1;min-width:200px}
.slider-wrap label{font-size:0.65rem;color:var(--muted);letter-spacing:1px;display:block;margin-bottom:10px}
input[type=range]{-webkit-appearance:none;width:100%;height:3px;background:var(--border2);border-radius:2px;outline:none;cursor:pointer}
input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:16px;height:16px;background:var(--rainy);border-radius:50%;border:3px solid var(--bg);box-shadow:0 0 10px rgba(56,189,248,0.6)}
.ll-box{background:var(--surface);border:1px solid var(--border);border-radius:6px;padding:14px 20px;min-width:200px}
.ll-box .lbl{font-size:0.58rem;letter-spacing:2px;color:var(--muted);text-transform:uppercase;margin-bottom:6px}
.ll-box .val{font-size:1.1rem;color:var(--green);font-weight:700}

.obs-row{display:flex;gap:10px;margin-bottom:28px;flex-wrap:wrap}
.obs-chip{font-size:0.75rem;letter-spacing:1.5px;text-transform:uppercase;padding:8px 18px;border-radius:4px;border:1px solid var(--border2);color:var(--muted);background:var(--surface)}
.obs-chip.Walk{border-color:var(--green);color:var(--green);background:rgba(74,222,128,0.08)}
.obs-chip.Shop{border-color:var(--rainy);color:var(--rainy);background:rgba(56,189,248,0.08)}
.obs-chip.Clean{border-color:var(--sunny);color:var(--sunny);background:rgba(251,191,36,0.08)}

.pi-pair{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.pi-item{background:var(--surface);border-radius:6px;padding:16px 20px;border-left:3px solid transparent}
.pi-item.rainy{border-color:var(--rainy)}
.pi-item.sunny{border-color:var(--sunny)}
.pi-item .name{font-size:0.6rem;letter-spacing:2px;text-transform:uppercase;color:var(--muted);margin-bottom:8px}
.pi-item .num{font-family:'Rajdhani',sans-serif;font-size:2rem;font-weight:700}
.pi-item.rainy .num{color:var(--rainy)}
.pi-item.sunny .num{color:var(--sunny)}

table{border-collapse:collapse;width:100%;font-size:0.78rem}
thead th{padding:8px 14px;text-align:center;color:var(--muted);font-size:0.6rem;letter-spacing:1.5px;text-transform:uppercase;border-bottom:1px solid var(--border);font-weight:400}
tbody td{padding:12px 14px;text-align:center;border-bottom:1px solid var(--border);transition:background 0.2s;position:relative}
tbody tr:last-child td{border-bottom:none}
tbody tr:hover td{background:rgba(56,189,248,0.04)}
.rl{text-align:left!important;font-weight:700;font-size:0.7rem;letter-spacing:1px}
.rl.r{color:var(--rainy)}.rl.s{color:var(--sunny)}
.cell-val{font-size:0.85rem;font-weight:500;position:relative;z-index:1}
.cell-bg{position:absolute;left:0;bottom:0;top:0;border-radius:2px;transition:width 0.5s ease;z-index:0}
.cell-bg.a{background:rgba(56,189,248,0.12)}
.cell-bg.b{background:rgba(251,191,36,0.12)}
.alpha-val{color:var(--rainy);font-weight:500}
.beta-val{color:var(--sunny);font-weight:500}

#state-svg{width:100%;height:240px;display:block}
#ll-canvas{width:100%;height:160px;display:block}

@keyframes fadeUp{from{opacity:0;transform:translateY(16px)}to{opacity:1;transform:translateY(0)}}
.card,.ctrl{animation:fadeUp 0.4s ease both}
</style>
</head>
<body>

<header>
  <h1>HMM <span>Baum-Welch</span></h1>
  <div class="sep"></div>
  <span class="tag">Pattern Recognition</span>
  <span class="tag">CSE S4</span>
  <div class="spacer"></div>
  <span class="live-ll" id="header-ll">LL: —</span>
</header>

<div class="page">

  <div class="sec">Observation Sequence</div>
  <div class="obs-row">
    <span class="obs-chip Walk">Walk</span>
    <span class="obs-chip Shop">Shop</span>
    <span class="obs-chip Clean">Clean</span>
    <span class="obs-chip Walk">Walk</span>
  </div>

  <div class="ctrl">
    <div>
      <div class="iter-big" id="iter-big">1</div>
      <div class="iter-lbl">Iteration</div>
    </div>
    <div class="slider-wrap">
      <label>DRAG TO SCRUB THROUGH TRAINING</label>
      <input type="range" id="iter-slider" min="1" max="15" value="1" oninput="goTo(+this.value)"/>
    </div>
    <div class="ll-box">
      <div class="lbl">Log-Likelihood</div>
      <div class="val" id="ll-val">—</div>
    </div>
  </div>

  <div class="row col2">
    <div class="card">
      <div class="clabel">Initial State Distribution — pi</div>
      <div class="pi-pair">
        <div class="pi-item rainy"><div class="name">Rainy</div><div class="num" id="pi0">—</div></div>
        <div class="pi-item sunny"><div class="name">Sunny</div><div class="num" id="pi1">—</div></div>
      </div>
    </div>
    <div class="card">
      <div class="clabel">State Transition Diagram</div>
      <svg id="state-svg" viewBox="0 0 500 220"></svg>
    </div>
  </div>

  <div class="row col2">
    <div class="card">
      <div class="clabel">Transition Matrix — A</div>
      <table><thead><tr><th></th><th>to Rainy</th><th>to Sunny</th></tr></thead>
      <tbody id="A-body"></tbody></table>
    </div>
    <div class="card">
      <div class="clabel">Emission Matrix — B</div>
      <table><thead><tr><th></th><th>Walk</th><th>Shop</th><th>Clean</th></tr></thead>
      <tbody id="B-body"></tbody></table>
    </div>
  </div>

  <div class="sec">Log-Likelihood Convergence</div>
  <div class="card" style="margin-bottom:28px"><canvas id="ll-canvas"></canvas></div>

  <div class="row col2">
    <div class="card">
      <div class="clabel">Forward Algorithm — alpha values</div>
      <table><thead><tr><th></th><th>t=0</th><th>t=1</th><th>t=2</th><th>t=3</th></tr></thead>
      <tbody id="alpha-body"></tbody></table>
    </div>
    <div class="card">
      <div class="clabel">Backward Algorithm — beta values</div>
      <table><thead><tr><th></th><th>t=0</th><th>t=1</th><th>t=2</th><th>t=3</th></tr></thead>
      <tbody id="beta-body"></tbody></table>
    </div>
  </div>

</div>

<script>
const H = HISTORY_PLACEHOLDER;

function fmt4(v){return v.toFixed(4)}
function fmtE(v){return v.toExponential(2)}

function goTo(n){
  const d=H[n-1];
  document.getElementById('iter-big').textContent=d.iteration;
  const ll=d.ll.toFixed(5);
  document.getElementById('ll-val').textContent=ll;
  document.getElementById('header-ll').textContent='LL: '+ll;
  document.getElementById('pi0').textContent=fmt4(d.pi[0]);
  document.getElementById('pi1').textContent=fmt4(d.pi[1]);

  document.getElementById('A-body').innerHTML=d.A.map((row,i)=>`
    <tr><td class="rl ${i===0?'r':'s'}">${i===0?'Rainy':'Sunny'}</td>
    ${row.map(v=>`<td><div class="cell-bg a" style="width:${(v*100).toFixed(0)}%"></div>
    <span class="cell-val" style="color:${i===0?'var(--rainy)':'var(--sunny)'}">${fmt4(v)}</span></td>`).join('')}</tr>`).join('');

  document.getElementById('B-body').innerHTML=d.B.map((row,i)=>`
    <tr><td class="rl ${i===0?'r':'s'}">${i===0?'Rainy':'Sunny'}</td>
    ${row.map(v=>`<td><div class="cell-bg b" style="width:${(v*100).toFixed(0)}%"></div>
    <span class="cell-val" style="color:var(--sunny)">${fmt4(v)}</span></td>`).join('')}</tr>`).join('');

  document.getElementById('alpha-body').innerHTML=[0,1].map(s=>`
    <tr><td class="rl ${s===0?'r':'s'}">${s===0?'Rainy':'Sunny'}</td>
    ${[0,1,2,3].map(t=>`<td class="alpha-val">${fmtE(d.alpha[t][s])}</td>`).join('')}</tr>`).join('');

  document.getElementById('beta-body').innerHTML=[0,1].map(s=>`
    <tr><td class="rl ${s===0?'r':'s'}">${s===0?'Rainy':'Sunny'}</td>
    ${[0,1,2,3].map(t=>`<td class="beta-val">${fmtE(d.beta[t][s])}</td>`).join('')}</tr>`).join('');

  drawDiagram(d.A,d.B);
  drawLL(n-1);
}

function drawDiagram(A,B){
  const svg=document.getElementById('state-svg');
  const cx=[130,370],cy=[110,110],r=60;
  svg.innerHTML=`
    <defs>
      <marker id="ar" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,1 L8,4 L0,7 Z" fill="#38bdf8"/></marker>
      <marker id="as" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,1 L8,4 L0,7 Z" fill="#fbbf24"/></marker>
      <marker id="ad" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,1 L8,4 L0,7 Z" fill="#4a6070"/></marker>
    </defs>
    <path d="M${cx[0]-28},${cy[0]-r} A38,38 0 1,1 ${cx[0]+28},${cy[0]-r}" fill="none" stroke="#38bdf8" stroke-width="2" marker-end="url(#ar)"/>
    <text x="${cx[0]}" y="${cy[0]-r-24}" text-anchor="middle" fill="#38bdf8" font-size="13" font-family="JetBrains Mono" font-weight="700">${A[0][0].toFixed(3)}</text>
    <path d="M${cx[1]-28},${cy[1]-r} A38,38 0 1,1 ${cx[1]+28},${cy[1]-r}" fill="none" stroke="#fbbf24" stroke-width="2" marker-end="url(#as)"/>
    <text x="${cx[1]}" y="${cy[1]-r-24}" text-anchor="middle" fill="#fbbf24" font-size="13" font-family="JetBrains Mono" font-weight="700">${A[1][1].toFixed(3)}</text>
    <path d="M${cx[0]+r},96 C${cx[0]+r+50},70 ${cx[1]-r-50},70 ${cx[1]-r},96" fill="none" stroke="#4a6070" stroke-width="1.8" marker-end="url(#ad)"/>
    <text x="250" y="60" text-anchor="middle" fill="#8aaabf" font-size="12" font-family="JetBrains Mono">${A[0][1].toFixed(3)}</text>
    <path d="M${cx[1]-r},124 C${cx[1]-r-50},152 ${cx[0]+r+50},152 ${cx[0]+r},124" fill="none" stroke="#4a6070" stroke-width="1.8" marker-end="url(#ad)"/>
    <text x="250" y="168" text-anchor="middle" fill="#8aaabf" font-size="12" font-family="JetBrains Mono">${A[1][0].toFixed(3)}</text>
    <circle cx="${cx[0]}" cy="${cy[0]}" r="${r}" fill="#0a1520" stroke="#38bdf8" stroke-width="2.5"/>
    <text x="${cx[0]}" y="${cy[0]-12}" text-anchor="middle" fill="#38bdf8" font-size="16" font-family="Rajdhani" font-weight="700" letter-spacing="1">RAINY</text>
    <text x="${cx[0]}" y="${cy[0]+7}" text-anchor="middle" fill="#5a8aaa" font-size="9.5" font-family="JetBrains Mono">W:${B[0][0].toFixed(2)}  S:${B[0][1].toFixed(2)}  C:${B[0][2].toFixed(2)}</text>
    <circle cx="${cx[1]}" cy="${cy[1]}" r="${r}" fill="#1a1000" stroke="#fbbf24" stroke-width="2.5"/>
    <text x="${cx[1]}" y="${cy[1]-12}" text-anchor="middle" fill="#fbbf24" font-size="16" font-family="Rajdhani" font-weight="700" letter-spacing="1">SUNNY</text>
    <text x="${cx[1]}" y="${cy[1]+7}" text-anchor="middle" fill="#aa8a4a" font-size="9.5" font-family="JetBrains Mono">W:${B[1][0].toFixed(2)}  S:${B[1][1].toFixed(2)}  C:${B[1][2].toFixed(2)}</text>
  `;
}

function drawLL(cur){
  const canvas=document.getElementById('ll-canvas');
  const dpr=window.devicePixelRatio||1;
  canvas.width=canvas.offsetWidth*dpr;
  canvas.height=canvas.offsetHeight*dpr;
  const ctx=canvas.getContext('2d');
  ctx.scale(dpr,dpr);
  const W=canvas.offsetWidth,CH=canvas.offsetHeight;
  const p={l:54,r:20,t:20,b:32};
  const lls=H.map(d=>d.ll);
  const mn=Math.min(...lls),mx=Math.max(...lls),rng=mx-mn||1;
  const px=i=>p.l+(W-p.l-p.r)*i/(lls.length-1);
  const py=v=>p.t+(CH-p.t-p.b)*(1-(v-mn)/rng);

  for(let i=0;i<=4;i++){
    const y=p.t+(CH-p.t-p.b)*i/4;
    ctx.strokeStyle='#1e2a36';ctx.lineWidth=1;
    ctx.beginPath();ctx.moveTo(p.l,y);ctx.lineTo(W-p.r,y);ctx.stroke();
    ctx.fillStyle='#3a5a6a';ctx.font='9px JetBrains Mono';ctx.textAlign='right';
    ctx.fillText((mx-rng*i/4).toFixed(2),p.l-6,y+3);
  }
  ctx.fillStyle='#3a5a6a';ctx.font='9px JetBrains Mono';ctx.textAlign='center';
  lls.forEach((_,i)=>{if(i%4===0||i===lls.length-1)ctx.fillText(i+1,px(i),CH-p.b+14);});

  ctx.beginPath();
  lls.forEach((v,i)=>{i===0?ctx.moveTo(px(i),py(v)):ctx.lineTo(px(i),py(v));});
  ctx.strokeStyle='#1e3040';ctx.lineWidth=2;ctx.stroke();

  ctx.beginPath();
  lls.slice(0,cur+1).forEach((v,i)=>{i===0?ctx.moveTo(px(i),py(v)):ctx.lineTo(px(i),py(v));});
  const g=ctx.createLinearGradient(0,0,W,0);
  g.addColorStop(0,'#38bdf8');g.addColorStop(1,'#4ade80');
  ctx.strokeStyle=g;ctx.lineWidth=2.5;ctx.stroke();

  ctx.beginPath();
  lls.slice(0,cur+1).forEach((v,i)=>{i===0?ctx.moveTo(px(i),py(v)):ctx.lineTo(px(i),py(v));});
  ctx.lineTo(px(cur),CH-p.b);ctx.lineTo(px(0),CH-p.b);ctx.closePath();
  const ag=ctx.createLinearGradient(0,p.t,0,CH-p.b);
  ag.addColorStop(0,'rgba(56,189,248,0.18)');ag.addColorStop(1,'rgba(56,189,248,0)');
  ctx.fillStyle=ag;ctx.fill();

  lls.slice(0,cur+1).forEach((v,i)=>{
    ctx.beginPath();ctx.arc(px(i),py(v),i===cur?5:3,0,Math.PI*2);
    ctx.fillStyle=i===cur?'#4ade80':'#38bdf8';
    if(i===cur){ctx.shadowColor='#4ade80';ctx.shadowBlur=12;}
    ctx.fill();ctx.shadowBlur=0;
  });
}

goTo(1);
window.addEventListener('resize',()=>{drawLL(+document.getElementById('iter-slider').value-1);});
</script>
</body>
</html>"""

html_content = html_content.replace("HISTORY_PLACEHOLDER", HISTORY_JSON)

tmp = tempfile.NamedTemporaryFile(
    mode='w', suffix='.html', delete=False,
    encoding='utf-8', prefix='hmm_visualizer_'
)
tmp.write(html_content)
tmp.close()

webbrowser.open('file://' + os.path.realpath(tmp.name))
print(f"  Browser opened -> {tmp.name}")
print("  Close terminal when done.\n")