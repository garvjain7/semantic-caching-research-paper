/* ==========================================================================
   app.js — Semantic Cache Research Dashboard
   ========================================================================== */

const API_BASE = 'http://localhost:8000';
const POLL_MS  = 2500;
let _pollTimer = null;

/* ── Scalar metric definitions for the Compare panel ───────────────────────
   key:     field name in summary.json
   label:   display name
   compare: "higher" | "lower" | null (null = no colouring, e.g. strings)
   ───────────────────────────────────────────────────────────────────────── */
const SCALAR_METRICS = [
  { key: 'strategy',      label: 'Strategy',          compare: null      },
  { key: 'threshold',     label: 'Threshold',          compare: null      },
  { key: 'total_queries', label: 'Total queries',      compare: null      },
  { key: 'hit_rate',      label: 'Hit rate',           compare: 'higher'  },
  { key: 'false_hit_rate',label: 'False-hit rate',     compare: 'lower'   },
  { key: 'precision',     label: 'Precision (TP/P)',   compare: 'higher'  },
  { key: 'recall',        label: 'Recall (TP/AP)',     compare: 'higher'  },
  { key: 'cost_savings',  label: 'Est. cost savings',  compare: 'higher'  },
];

/* ==========================================================================
   Tab switching
   ========================================================================== */
function switchTab(name) {
  document.querySelectorAll('.tab-btn').forEach(b => {
    const active = b.id === `tab-${name}-btn`;
    b.classList.toggle('active', active);
    b.setAttribute('aria-selected', active);
  });
  document.querySelectorAll('.tab-panel').forEach(p =>
    p.classList.toggle('active', p.id === `tab-${name}`)
  );
}

/* ==========================================================================
   API connectivity check (polled every 8 s)
   ========================================================================== */
async function checkApi() {
  const dot  = document.getElementById('api-dot');
  const text = document.getElementById('api-status-text');
  try {
    const r = await fetch(`${API_BASE}/runs`, { signal: AbortSignal.timeout(3000) });
    if (r.ok) {
      dot.className    = 'status-dot connected';
      text.textContent = 'API online';
      return;
    }
  } catch (_) {}
  dot.className    = 'status-dot error';
  text.textContent = 'API offline — start uvicorn';
}

/* ==========================================================================
   Console helper (Run tab log pane)
   ========================================================================== */
function logToConsole(msg, level = 'dim') {
  const c  = document.getElementById('run-console');
  const ts = new Date().toLocaleTimeString();
  const el = document.createElement('span');
  el.className   = `log-line ${level}`;
  el.textContent = `[${ts}]  ${msg}`;
  // Clear the placeholder text on first real message
  if (c.querySelector('.log-line.dim')?.textContent?.includes('no output'))
    c.innerHTML = '';
  c.appendChild(el);
  c.appendChild(document.createElement('br'));
  c.scrollTop = c.scrollHeight;
}

/* ==========================================================================
   Run tab — start a new experiment run
   ========================================================================== */
async function startRun() {
  const config   = document.getElementById('select-config').value;
  const strategy = document.getElementById('select-strategy').value;
  const runId    = document.getElementById('input-run-id').value.trim() || null;
  const btn      = document.getElementById('btn-start-run');

  btn.disabled = true;
  logToConsole(`POST /run  config=${config}  strategy=${strategy}`, 'info');

  try {
    const resp = await fetch(`${API_BASE}/run`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ config, match_strategy: strategy, run_id: runId }),
    });

    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ detail: resp.statusText }));
      logToConsole(`Error ${resp.status}: ${err.detail}`, 'err');
      btn.disabled = false;
      return;
    }

    const data = await resp.json();
    logToConsole(`Run started — ID: ${data.run_id}`, 'ok');
    _showRunCard(data.run_id, 'queued', 0, 0);
    _startPolling(data.run_id);
  } catch (e) {
    logToConsole(`Could not reach API: ${e.message}`, 'err');
    btn.disabled = false;
  }
}

function _showRunCard(runId, status, progress, total) {
  document.getElementById('active-run-card').style.display = 'block';
  document.getElementById('active-run-id').textContent     = runId;
  _updateProgress(status, progress, total);
}

function _updateProgress(status, progress, total) {
  const badge = document.getElementById('active-run-badge');
  const bar   = document.getElementById('active-progress-bar');
  const label = document.getElementById('active-progress-label');

  badge.className   = `badge badge-${status}`;
  badge.textContent = status;

  const pct = total > 0 ? Math.round((progress / total) * 100) : 0;
  bar.style.width   = `${pct}%`;
  label.textContent = total > 0 ? `${progress} / ${total} queries (${pct}%)` : status;
}

/* ==========================================================================
   Status polling
   ========================================================================== */
function _startPolling(runId) {
  clearInterval(_pollTimer);
  _pollTimer = setInterval(() => _pollStatus(runId), POLL_MS);
}

async function _pollStatus(runId) {
  try {
    const r = await fetch(`${API_BASE}/status/${runId}`);
    if (!r.ok) return;
    const d = await r.json();
    _updateProgress(d.status, d.progress, d.total);

    if (d.status === 'done') {
      logToConsole(`Run ${runId} complete ✓`, 'ok');
      document.getElementById('btn-start-run').disabled = false;
      clearInterval(_pollTimer);
    } else if (d.status === 'error') {
      logToConsole(`Run ${runId} errored: ${d.error}`, 'err');
      document.getElementById('btn-start-run').disabled = false;
      clearInterval(_pollTimer);
    }
  } catch (_) { /* network blip — retry next tick */ }
}

/* ==========================================================================
   Compare tab — load and render two run summaries side by side
   ========================================================================== */
async function compareRuns() {
  const runA = document.getElementById('compare-run-a').value.trim();
  const runB = document.getElementById('compare-run-b').value.trim();
  if (!runA || !runB) { alert('Please enter both run IDs.'); return; }

  const hint = document.getElementById('compare-hint');
  const btn  = document.getElementById('btn-compare');
  hint.textContent = 'Loading…';
  btn.disabled = true;

  try {
    const url = `${API_BASE}/compare?run_a=${encodeURIComponent(runA)}&run_b=${encodeURIComponent(runB)}`;
    const r   = await fetch(url);

    if (!r.ok) {
      const err = await r.json().catch(() => ({ detail: r.statusText }));
      hint.textContent = `Error: ${err.detail}`;
      btn.disabled = false;
      return;
    }

    const { run_a, run_b } = await r.json();
    _renderCompare('a', runA, run_a, run_b);
    _renderCompare('b', runB, run_b, run_a);
    document.getElementById('compare-results').style.display = 'block';
    hint.textContent = '';
  } catch (e) {
    hint.textContent = `Could not reach API: ${e.message}`;
  }
  btn.disabled = false;
}

/* Format a single metric value for display */
function _fmtVal(v) {
  if (typeof v === 'number') return (v > 0 && v < 1) ? `${(v * 100).toFixed(1)}%` : String(v);
  return v ?? '—';
}

/* Render one column (side = "a" or "b") of the compare grid */
function _renderCompare(side, runId, me, other) {
  document.getElementById(`compare-title-${side}`).textContent = `Run: ${runId}`;
  const el = document.getElementById(`compare-metrics-${side}`);

  /* --- Scalar metric rows --- */
  const rows = SCALAR_METRICS.map(({ key, label, compare }) => {
    const v = me[key], ov = other[key];
    let cls = '';
    if (compare && typeof v === 'number' && typeof ov === 'number') {
      const better = compare === 'higher' ? v > ov : v < ov;
      const worse  = compare === 'higher' ? v < ov : v > ov;
      cls = better ? 'better' : worse ? 'worse' : '';
    }
    return `<div class="metric-row">
      <span class="metric-name">${label}</span>
      <span class="metric-val ${cls}">${_fmtVal(v)}</span>
    </div>`;
  }).join('');

  /* --- Confusion matrix --- */
  const cm = me.confusion_matrix || {};
  const cmHtml = `
    <hr class="section-sep">
    <div class="card-title" style="margin-top:0.5rem">Confusion matrix</div>
    <div class="cm-grid">
      <div class="cm-cell cm-tp">
        <span class="cm-label">TP — hit ∩ paraphrase</span>
        <span class="cm-value">${cm.TP ?? '—'}</span>
      </div>
      <div class="cm-cell cm-fp">
        <span class="cm-label">FP — hit ∩ non-paraphrase</span>
        <span class="cm-value">${cm.FP ?? '—'}</span>
      </div>
      <div class="cm-cell cm-fn">
        <span class="cm-label">FN — miss ∩ paraphrase</span>
        <span class="cm-value">${cm.FN ?? '—'}</span>
      </div>
      <div class="cm-cell cm-tn">
        <span class="cm-label">TN — miss ∩ non-paraphrase</span>
        <span class="cm-value">${cm.TN ?? '—'}</span>
      </div>
    </div>`;

  el.innerHTML = rows + cmHtml;
}

/* ==========================================================================
   Initialise on page load
   ========================================================================== */
document.addEventListener('DOMContentLoaded', () => {
  checkApi();
  setInterval(checkApi, 8000);
});
