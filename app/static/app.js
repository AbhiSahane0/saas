const $ = s => document.querySelector(s);
const COLORS = {joy:'#f5c542',sadness:'#4da3ff',anger:'#ff6b6b',fear:'#a78bfa',love:'#f472b6',surprise:'#34d399'};
const pill = e => `<span class="pill" style="background:${COLORS[e]}">${e}</span>`;
const pct = x => (x * 100).toFixed(1) + '%';
const esc = s => s.replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
let poll, charts = {}, results = [], sortKey = 'accuracy', es;
Chart.defaults.color = '#96a3c8'; Chart.defaults.borderColor = '#262d55'; Chart.defaults.animation = false;

async function api(url, opts) {
  const r = await fetch(url, opts);
  if (r.status === 401) { showLogin(); throw new Error('login'); }
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.detail || r.statusText);
  return j;
}
function showLogin() { $('#app').hidden = true; $('#login').hidden = false; stopPolling(); }
function showApp() { $('#login').hidden = true; $('#app').hidden = false; refresh(); }
// poll /api/status only while a training run is in progress; stop as soon as it finishes
function startPolling() { if (!poll) poll = setInterval(refresh, 1500); }
function stopPolling() { clearInterval(poll); poll = null; }

$('#loginForm').onsubmit = async e => {
  e.preventDefault(); $('#loginErr').textContent = '';
  try { await api('/api/login', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({username:$('#user').value, password:$('#pass').value})}); showApp(); }
  catch (err) { $('#loginErr').textContent = err.message; }
};
$('#logout').onclick = async () => { await fetch('/api/logout', {method:'POST'}); if (es) es.close(); showLogin(); };
document.querySelectorAll('nav button').forEach(b => b.onclick = () => {
  document.querySelectorAll('nav button').forEach(x => x.classList.toggle('active', x === b));
  document.querySelectorAll('.tab').forEach(t => t.hidden = t.id !== 'tab-' + b.dataset.tab);
  Object.values(charts).forEach(c => c.resize());
});

$('#runBtn').onclick = async () => {
  $('#runErr').textContent = '';
  const fd = new FormData(); const f = $('#file').files[0]; if (f) { fd.append('file', f); fd.append('include_lstm', $('#lstm').checked); }
  try { await api('/api/run', {method:'POST', body: fd}); refresh(); } catch (e) { $('#runErr').textContent = e.message; }
};

async function refresh() {
  let s; try { s = await api('/api/status'); } catch { return stopPolling(); }
  if (s.state === 'running') startPolling(); else stopPolling();
  $('#barFill').style.width = (s.state === 'done' ? 100 : s.progress * 100) + '%';
  $('#statusMsg').textContent = s.state === 'done' ? `Done – ${s.results.length} experiments complete.` :
    s.state === 'error' ? '' : s.message;
  if (s.state === 'error') $('#runErr').textContent = s.error;
  $('#runBtn').disabled = s.state === 'running'; $('#predictBtn').disabled = s.state !== 'done'; $('#streamBtn').disabled = s.state !== 'done';
  if (s.state === 'done' && JSON.stringify(s.results) !== JSON.stringify(results)) { results = s.results; render(s.summary); }
}

function card(v, l) { return `<div class="c"><b>${v}</b><span>${l}</span></div>`; }
function render(sum) {
  $('#summaryCards').innerHTML = card(sum.total, 'Total tweets') + card(sum.train, 'Train') + card(sum.test, 'Test (held out)');
  const best = results.reduce((a, b) => b.accuracy > a.accuracy ? b : a);
  $('#bestCards').innerHTML = card(best.classifier, 'Best model') + card(pct(best.accuracy), 'Accuracy') + card(pct(best.f1), 'Macro F1') +
    card(best.mode, 'Feature mode');
  draw('pie', {type:'doughnut', data:{labels:Object.keys(sum.distribution), datasets:[{data:Object.values(sum.distribution),
    backgroundColor:Object.keys(sum.distribution).map(k => COLORS[k]), borderWidth:0}]}, options:{maintainAspectRatio:false,
    plugins:{legend:{position:'right'}}}});
  const modes = [...new Set(results.map(r => r.mode))], clfs = [...new Set(results.map(r => r.classifier))];
  const pal = ['#4da3ff','#3ecf8e','#f5c542','#f472b6','#a78bfa'];
  draw('bar', {type:'bar', data:{labels:modes, datasets:clfs.map((c, i) => ({label:c, backgroundColor:pal[i % 5],
    data:modes.map(m => +(results.find(r => r.classifier === c && r.mode === m).accuracy * 100).toFixed(1))}))},
    options:{maintainAspectRatio:false, scales:{y:{beginAtZero:true, max:100, title:{display:true, text:'Accuracy %'}}}}});
  table();
}
function draw(id, cfg) { if (charts[id]) charts[id].destroy(); charts[id] = new Chart($('#' + id), cfg); }

function table() {
  const cols = [['classifier','Classifier'],['mode','Features'],['accuracy','Accuracy'],['precision','Precision'],['recall','Recall'],
    ['f1','F1'],['train_ms','Train ms'],['infer_ms','Infer ms']];
  const rows = [...results].sort((a, b) => typeof a[sortKey] === 'string' ? a[sortKey].localeCompare(b[sortKey]) : b[sortKey] - a[sortKey]);
  $('#resTable').innerHTML = `<thead><tr>${cols.map(([k, n]) => `<th data-k="${k}">${n}${k === sortKey ? ' ▼' : ''}</th>`).join('')}</tr></thead><tbody>` +
    rows.map((r, i) => `<tr data-i="${results.indexOf(r)}"><td>${r.classifier}</td><td>${r.mode}</td><td>${pct(r.accuracy)}</td><td>${pct(r.precision)}</td>` +
      `<td>${pct(r.recall)}</td><td>${pct(r.f1)}</td><td>${r.train_ms}</td><td>${r.infer_ms}</td></tr>`).join('') + '</tbody>';
  $('#resTable').querySelectorAll('th').forEach(th => th.onclick = () => { sortKey = th.dataset.k; table(); });
  $('#resTable').querySelectorAll('tbody tr').forEach(tr => tr.onclick = () => {
    $('#resTable').querySelectorAll('tr').forEach(x => x.classList.remove('sel')); tr.classList.add('sel'); detail(results[tr.dataset.i]); });
}
function detail(r) {
  $('#detail').hidden = false; $('#detailTitle').textContent = `${r.classifier} · ${r.mode} — confusion matrix (rows = actual, columns = predicted)`;
  const L = Object.keys(COLORS), mx = Math.max(...r.confusion.flat());
  $('#cm').innerHTML = `<div class="cm" style="grid-template-columns:70px repeat(6,1fr)"><div class="h"></div>${L.map(l => `<div class="h">${l}</div>`).join('')}` +
    r.confusion.map((row, i) => `<div class="h">${L[i]}</div>` + row.map((v, j) =>
      `<div style="background:rgba(77,163,255,${(v / mx * .9).toFixed(2)});${i === j ? 'outline:1px solid #3ecf8e' : ''}">${v}</div>`).join('')).join('') + '</div>';
  $('#pcTable').innerHTML = '<thead><tr><th>Emotion</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr></thead><tbody>' +
    r.per_class.map(c => `<tr><td>${pill(c.label)}</td><td>${pct(c.precision)}</td><td>${pct(c.recall)}</td><td>${pct(c.f1)}</td><td>${c.support}</td></tr>`).join('') + '</tbody>';
  $('#detail').scrollIntoView({behavior:'smooth'});
}

async function predict() {
  const text = $('#text').value.trim(); if (!text) return;
  try {
    const p = await api('/api/predict', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({text})});
    $('#predOut').innerHTML = `<p>Best model (<b>${p.best_model}</b>) says: ${pill(p.consensus)}</p><div class="preds">` +
      Object.entries(p.predictions).map(([n, v]) => `<div class="c"><span>${n} · ${v.mode} · ${pct(v.accuracy)} acc</span><b>${v.emotion}</b></div>`).join('') + '</div>';
  } catch (e) { $('#predOut').innerHTML = `<p class="err">${e.message}</p>`; }
}
$('#predictBtn').onclick = predict; $('#text').onkeydown = e => { if (e.key === 'Enter') predict(); };

function stopStream() { if (es) es.close(); es = null; $('#streamBtn').textContent = 'Start stream'; }
$('#streamBtn').onclick = () => {
  if (es) { stopStream(); return; }
  const source = $('#source').value, q = $('#keyword').value.trim();
  $('#feed').innerHTML = ''; $('#streamStat').textContent = source === 'bluesky' ? 'Connecting to Bluesky…' : 'Starting…';
  let n = 0, ok = 0, rated = 0, right = 0; const tally = Object.fromEntries(Object.keys(COLORS).map(k => [k, 0]));
  es = new EventSource(`/api/stream?source=${source}&q=${encodeURIComponent(q)}`); $('#streamBtn').textContent = 'Stop stream';
  es.onmessage = m => {
    const d = JSON.parse(m.data);
    if (d.notice) { $('#streamStat').textContent = d.notice; return; }
    n++; tally[d.predicted]++;
    const base = `model: ${d.model} (${d.mode}) · ${pct(d.accuracy)} accuracy on held-out test set`;
    const stat = () => {
      const r = rated ? ` · your ratings: ${right}/${rated} correct (${pct(right / rated)})` : '';
      $('#streamStat').textContent = `${n} classified · ` + (d.actual ? `${pct(ok / n)} matched the dataset label · ` : '') + base + r;
    };
    if (d.actual && d.predicted === d.actual) ok++;
    stat();
    $('#tally').innerHTML = Object.entries(tally).map(([k, v]) => `${pill(k)} ${v}`).join(' &nbsp; ');
    const mark = d.actual ? `<span class="${d.predicted === d.actual ? 'ok' : 'no'}">${d.predicted === d.actual ? '✓' : '✗'}</span>`
      : `<span class="rate"><button data-r="1" title="Prediction is correct">✓</button><button data-r="0" title="Prediction is wrong">✗</button></span>`;
    const sub = d.actual ? `labelled: ${d.actual}` : `<a href="${d.url}" target="_blank" rel="noopener noreferrer">view on Bluesky</a>`;
    $('#feed').insertAdjacentHTML('afterbegin', `<div class="item"><div>${esc(d.text)}<br><small>${sub}</small></div><div>${pill(d.predicted)} ${mark}</div></div>`);
    const rate = $('#feed .item .rate');
    if (rate) rate.onclick = e => {
      const b = e.target.closest('button'); if (!b) return;
      rated++; right += +b.dataset.r;
      rate.innerHTML = b.dataset.r === '1' ? '<span class="ok">✓ correct</span>' : '<span class="no">✗ wrong</span>';
      rate.onclick = null; stat();
    };
    while ($('#feed').children.length > 100) $('#feed').lastChild.remove();
  };
  es.onerror = () => { if (es && es.readyState === EventSource.CLOSED) stopStream(); };
};
fetch('/api/me').then(r => r.ok ? showApp() : showLogin());
