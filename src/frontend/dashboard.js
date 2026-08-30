/**
 * SmartSpend AI - Dashboard Frontend JavaScript
 * Handles API data fetching, Chart.js visualizations, 7x24 Heatmap rendering,
 * searchable/paginated transactions table, and live AI multi-model inference.
 */

// Application state
let appState = {
  nudgeThreshold: 70,
  currentPage: 0,
  pageSize: 25,
  totalCount: 0,
  filteredCount: 0,
  selectedCategory: 'all',
  searchQuery: '',
  isNudgeOnly: false,
  charts: {}
};

// Preset sample transactions for rapid live testing
const PRESETS = {
  'late-food': {
    merchant: 'GrabFood',
    memo: 'สั่งพิซซ่ามื้อดึกหิวมากหลังเที่ยงคืน',
    amount: '420.00',
    date: '2025-01-26', // Payday period
    time: '23:45'       // Late night
  },
  'lunch': {
    merchant: 'ร้านก๋วยเตี๋ยวป้าเพ็ญ',
    memo: 'ข้าวเที่ยงวันทำงาน',
    amount: '65.00',
    date: '2025-02-10',
    time: '12:15'
  },
  'shopee-payday': {
    merchant: 'Shopee Official Store',
    memo: 'กดสั่งรองเท้าผ้าใบโปร Flash Sale',
    amount: '2150.00',
    date: '2025-01-27', // Payday period
    time: '14:30'
  }
};

document.addEventListener('DOMContentLoaded', async () => {
  setupEventListeners();
  await initializeDashboard();
});

function setupEventListeners() {
  // Refresh Button
  const btnRefresh = document.getElementById('btn-refresh');
  if (btnRefresh) {
    btnRefresh.addEventListener('click', async () => {
      btnRefresh.classList.add('animate-spin');
      await initializeDashboard();
      setTimeout(() => btnRefresh.classList.remove('animate-spin'), 600);
    });
  }

  // Preset Buttons
  document.querySelectorAll('.btn-preset').forEach(btn => {
    btn.addEventListener('click', (e) => {
      const presetKey = e.currentTarget.dataset.preset;
      const data = PRESETS[presetKey];
      if (data) {
        document.getElementById('input-merchant').value = data.merchant;
        document.getElementById('input-memo').value = data.memo;
        document.getElementById('input-amount').value = data.amount;
        document.getElementById('input-date').value = data.date;
        document.getElementById('input-time').value = data.time;
      }
    });
  });

  // Live Predict Form Submission
  const predictForm = document.getElementById('predict-form');
  if (predictForm) {
    predictForm.addEventListener('submit', handleLivePrediction);
  }

  // Transaction Table Search & Filters
  const searchInput = document.getElementById('table-search');
  if (searchInput) {
    let debounceTimer;
    searchInput.addEventListener('input', (e) => {
      clearTimeout(debounceTimer);
      debounceTimer = setTimeout(() => {
        appState.searchQuery = e.target.value;
        appState.currentPage = 0;
        loadTransactions();
      }, 300);
    });
  }

  const categoryFilter = document.getElementById('table-category-filter');
  if (categoryFilter) {
    categoryFilter.addEventListener('change', (e) => {
      appState.selectedCategory = e.target.value;
      appState.currentPage = 0;
      loadTransactions();
    });
  }

  const toggleNudge = document.getElementById('toggle-nudge-only');
  if (toggleNudge) {
    toggleNudge.addEventListener('change', (e) => {
      appState.isNudgeOnly = e.target.checked;
      appState.currentPage = 0;
      loadTransactions();
    });
  }

  // Pagination Buttons
  const btnPrev = document.getElementById('btn-prev-page');
  const btnNext = document.getElementById('btn-next-page');
  if (btnPrev) {
    btnPrev.addEventListener('click', () => {
      if (appState.currentPage > 0) {
        appState.currentPage--;
        loadTransactions();
      }
    });
  }
  if (btnNext) {
    btnNext.addEventListener('click', () => {
      const maxPage = Math.ceil(appState.filteredCount / appState.pageSize) - 1;
      if (appState.currentPage < maxPage) {
        appState.currentPage++;
        loadTransactions();
      }
    });
  }
}

async function initializeDashboard() {
  try {
    // 1. Fetch config/metrics to get dynamic nudge_threshold
    await loadMetricsConfig();

    // 2. Fetch summary & render KPI cards + charts
    await loadSummary();

    // 3. Fetch & render spending heatmap
    await loadHeatmap();

    // 4. Fetch initial page of transactions
    await loadTransactions();

    // Set default date & time in form to now
    const now = new Date();
    const todayStr = now.toISOString().split('T')[0];
    const timeStr = `${String(now.getHours()).padStart(2, '0')}:${String(now.getMinutes()).padStart(2, '0')}`;
    const dateInput = document.getElementById('input-date');
    const timeInput = document.getElementById('input-time');
    if (dateInput && !dateInput.value) dateInput.value = todayStr;
    if (timeInput && !timeInput.value) timeInput.value = timeStr;

  } catch (err) {
    console.error('[Dashboard Init Error]', err);
  }
}

/**
 * Loads configuration metrics to retrieve dynamic nudge_threshold (zero hardcoding).
 */
async function loadMetricsConfig() {
  try {
    const res = await fetch('/api/metrics');
    if (res.ok) {
      const data = await res.json();
      appState.nudgeThreshold = data.nudge_threshold || 70;
      const elThresh = document.getElementById('kpi-nudge-thresh');
      if (elThresh) elThresh.textContent = appState.nudgeThreshold;
    }
  } catch (e) {
    console.warn('Could not fetch metrics config, using default 70', e);
  }
}

/**
 * Loads KPI Summary and renders Doughnut + Monthly Trend Charts.
 */
async function loadSummary() {
  const res = await fetch('/api/summary');
  if (!res.ok) throw new Error('Failed to load summary');
  const data = await res.json();

  // 1. Populate KPI Cards
  document.getElementById('kpi-total-spend').textContent = `฿${data.total_spend.toLocaleString('en-US', { minimumFractionDigits: 2 })}`;
  document.getElementById('kpi-total-tx').textContent = data.total_transactions.toLocaleString();
  
  document.getElementById('kpi-needs-pct').textContent = `${data.needs_percentage}%`;
  document.getElementById('kpi-wants-pct').textContent = `${data.wants_percentage}%`;
  document.getElementById('kpi-needs-amt').textContent = `฿${data.needs_amount.toLocaleString()}`;
  document.getElementById('kpi-wants-amt').textContent = `฿${data.wants_amount.toLocaleString()}`;

  document.getElementById('kpi-impulse-amt').textContent = `฿${data.impulse_spending_amount.toLocaleString('en-US', { minimumFractionDigits: 2 })}`;
  document.getElementById('kpi-impulse-pct').textContent = `${data.impulse_spending_percentage}%`;
  document.getElementById('kpi-impulse-count').textContent = `${data.impulse_transactions_count} รายการ`;

  document.getElementById('kpi-nudge-count').textContent = data.nudge_alerts_count.toLocaleString();

  // Legends
  document.getElementById('legend-needs-val').textContent = `${data.needs_percentage}% (฿${data.needs_amount.toLocaleString()})`;
  document.getElementById('legend-wants-val').textContent = `${data.wants_percentage}% (฿${data.wants_amount.toLocaleString()})`;

  // 2. Render Doughnut Chart (Needs vs Wants)
  renderNeedsWantsChart(data.needs_amount, data.wants_amount);

  // 3. Render Monthly Trend Chart
  renderMonthlyTrendChart(data.monthly_trend);
}

function renderNeedsWantsChart(needsAmt, wantsAmt) {
  const ctx = document.getElementById('chart-needs-wants').getContext('2d');
  if (appState.charts.needsWants) {
    appState.charts.needsWants.destroy();
  }

  appState.charts.needsWants = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: ['Needs (จำเป็น)', 'Wants (ฟุ่มเฟือย)'],
      datasets: [{
        data: [needsAmt, wantsAmt],
        backgroundColor: ['#10b981', '#f59e0b'],
        borderColor: '#0f172a',
        borderWidth: 3,
        hoverOffset: 6
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: '70%',
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: function(context) {
              const val = context.raw;
              const total = needsAmt + wantsAmt;
              const pct = ((val / total) * 100).toFixed(1);
              return ` ${context.label}: ฿${val.toLocaleString()} (${pct}%)`;
            }
          }
        }
      }
    }
  });
}

function renderMonthlyTrendChart(monthlyTrends) {
  const ctx = document.getElementById('chart-monthly-trend').getContext('2d');
  if (appState.charts.monthlyTrend) {
    appState.charts.monthlyTrend.destroy();
  }

  const labels = monthlyTrends.map(m => m.month);
  const totalAmounts = monthlyTrends.map(m => m.total_amount);
  const impulseAmounts = monthlyTrends.map(m => m.impulse_amount);

  appState.charts.monthlyTrend = new Chart(ctx, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [
        {
          label: 'ยอดใช้จ่ายรวม (Total Spend)',
          data: totalAmounts,
          borderColor: '#6366f1',
          backgroundColor: 'rgba(99, 102, 241, 0.1)',
          fill: true,
          tension: 0.35,
          borderWidth: 2,
          pointRadius: 3,
          pointBackgroundColor: '#6366f1'
        },
        {
          label: 'ยอด Impulse Buying',
          data: impulseAmounts,
          borderColor: '#ec4899',
          backgroundColor: 'rgba(236, 72, 153, 0.15)',
          fill: true,
          tension: 0.35,
          borderWidth: 2,
          pointRadius: 4,
          pointBackgroundColor: '#ec4899'
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          grid: { color: 'rgba(51, 65, 85, 0.3)' },
          ticks: { color: '#94a3b8', font: { size: 10 } }
        },
        y: {
          grid: { color: 'rgba(51, 65, 85, 0.3)' },
          ticks: {
            color: '#94a3b8',
            font: { size: 10 },
            callback: value => `฿${(value / 1000).toFixed(0)}k`
          }
        }
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: context => ` ${context.dataset.label}: ฿${context.raw.toLocaleString()}`
          }
        }
      }
    }
  });
}

/**
 * Fetches and renders the 7 Days x 24 Hours Spending & Impulse Heatmap.
 */
async function loadHeatmap() {
  const res = await fetch('/api/heatmap');
  if (!res.ok) return;
  const data = await res.json();

  const container = document.getElementById('heatmap-container');
  if (!container) return;

  const days = data.days;
  const hours = data.hours;
  const matrix = data.matrix;
  const maxAmt = data.max_amount_cell || 1;

  let html = `
    <div class="grid grid-cols-[80px_repeat(24,minmax(0,1fr))] gap-1 text-center text-[10px]">
      <!-- Header: Hours 00 to 23 -->
      <div class="py-1 text-slate-500 font-medium">วัน \ ชม.</div>
  `;

  for (let h = 0; h < 24; h++) {
    const isLate = (h >= 23 || h <= 2);
    html += `<div class="py-1 font-mono ${isLate ? 'text-pink-400 font-bold' : 'text-slate-400'}">${String(h).padStart(2, '0')}</div>`;
  }

  // Rows for each Day
  for (let d = 0; d < 7; d++) {
    html += `<div class="text-left font-medium text-slate-300 py-1.5 truncate pr-1">${days[d].split(' ')[0]}</div>`;
    for (let h = 0; h < 24; h++) {
      const cell = matrix[d][h];
      const amt = cell.total_amount;
      const count = cell.transaction_count;
      const imp = cell.impulse_count;
      const isLate = cell.is_late_night;

      // Color Intensity
      let bgClass = 'bg-slate-900/90 border-slate-800';
      if (amt > 0) {
        const ratio = amt / maxAmt;
        if (imp > 0 && isLate) {
          bgClass = 'bg-gradient-to-tr from-pink-600 to-rose-500 text-white shadow-sm shadow-pink-500/20';
        } else if (ratio > 0.6) {
          bgClass = 'bg-indigo-600 text-white';
        } else if (ratio > 0.3) {
          bgClass = 'bg-indigo-800/90 text-indigo-100';
        } else if (ratio > 0.1) {
          bgClass = 'bg-indigo-950/80 text-indigo-300';
        } else {
          bgClass = 'bg-slate-900/80 text-slate-400';
        }
      }

      html += `
        <div class="heatmap-cell relative rounded h-7 flex items-center justify-center cursor-pointer border ${bgClass}"
             title="${days[d]} ${String(h).padStart(2, '0')}:00\nยอดเงิน: ฿${amt.toLocaleString()}\nธุรกรรม: ${count} รายการ (Impulse: ${imp})">
          ${count > 0 ? `<span class="text-[9px] font-semibold">${count}</span>` : ''}
        </div>
      `;
    }
  }

  html += `</div>`;
  container.innerHTML = html;
}

/**
 * Fetches and populates the paginated and filtered transactions table.
 */
async function loadTransactions() {
  const skip = appState.currentPage * appState.pageSize;
  let url = `/api/transactions?limit=${appState.pageSize}&skip=${skip}`;

  if (appState.selectedCategory && appState.selectedCategory !== 'all') {
    url += `&category=${encodeURIComponent(appState.selectedCategory)}`;
  }
  if (appState.searchQuery) {
    url += `&search=${encodeURIComponent(appState.searchQuery)}`;
  }
  if (appState.isNudgeOnly) {
    url += `&is_nudge_only=true`;
  }

  const res = await fetch(url);
  if (!res.ok) return;
  const data = await res.json();

  appState.totalCount = data.total_count;
  appState.filteredCount = data.filtered_count;

  // Render Table Rows
  const tbody = document.getElementById('transactions-table-body');
  if (!tbody) return;

  if (data.items.length === 0) {
    tbody.innerHTML = `
      <tr>
        <td colspan="7" class="px-4 py-8 text-center text-slate-500">
          ไม่พบรายการที่ตรงกับเงื่อนไขการค้นหา
        </td>
      </tr>
    `;
  } else {
    tbody.innerHTML = data.items.map(tx => {
      const categoryColor = getCategoryColor(tx.category);
      const isNudge = tx.is_nudge_alert;

      return `
        <tr class="hover:bg-slate-900/60 transition ${isNudge ? 'bg-rose-950/10' : ''}">
          <td class="px-4 py-3 font-mono text-slate-400 whitespace-nowrap">
            <div>${tx.date}</div>
            <div class="text-[10px] text-slate-500">${tx.time}</div>
          </td>
          <td class="px-4 py-3">
            <div class="font-semibold text-white">${escapeHtml(tx.merchant)}</div>
            ${tx.memo ? `<div class="text-[11px] text-slate-400 truncate max-w-xs">${escapeHtml(tx.memo)}</div>` : ''}
          </td>
          <td class="px-4 py-3 text-right font-mono font-semibold text-white whitespace-nowrap">
            ฿${tx.amount.toLocaleString('en-US', { minimumFractionDigits: 2 })}
          </td>
          <td class="px-4 py-3">
            <span class="px-2.5 py-0.5 rounded-full text-[10px] font-semibold ${categoryColor}">
              ${escapeHtml(tx.category)}
            </span>
          </td>
          <td class="px-4 py-3 text-center">
            <span class="px-2 py-0.5 rounded text-[10px] font-medium ${tx.is_wants ? 'bg-amber-500/10 text-amber-400 border border-amber-500/20' : 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'}">
              ${tx.is_wants ? 'Wants' : 'Needs'}
            </span>
          </td>
          <td class="px-4 py-3 text-center">
            <span class="font-mono font-bold ${tx.impulse_score >= appState.nudgeThreshold ? 'text-rose-400' : (tx.impulse_score >= 40 ? 'text-amber-400' : 'text-slate-300')}">
              ${tx.impulse_score}
            </span>
          </td>
          <td class="px-4 py-3 text-center whitespace-nowrap">
            ${isNudge ? 
              `<span class="px-2 py-0.5 rounded-full bg-rose-500/20 text-rose-300 border border-rose-500/30 text-[10px] font-bold animate-pulse">🚨 Nudge</span>` : 
              `<span class="text-slate-600">-</span>`
            }
          </td>
        </tr>
      `;
    }).join('');
  }

  // Update Pagination Controls
  const startIdx = data.filtered_count === 0 ? 0 : skip + 1;
  const endIdx = Math.min(skip + data.items.length, data.filtered_count);
  document.getElementById('page-start-idx').textContent = startIdx.toLocaleString();
  document.getElementById('page-end-idx').textContent = endIdx.toLocaleString();
  document.getElementById('page-total-tx').textContent = data.filtered_count.toLocaleString();

  const btnPrev = document.getElementById('btn-prev-page');
  const btnNext = document.getElementById('btn-next-page');
  if (btnPrev) btnPrev.disabled = (appState.currentPage === 0);
  if (btnNext) btnNext.disabled = (endIdx >= data.filtered_count);
}

/**
 * Handles Live AI Simulation Form submission.
 */
async function handleLivePrediction(e) {
  e.preventDefault();

  const btnSubmit = document.getElementById('btn-submit-predict');
  const originalBtnText = btnSubmit.innerHTML;
  btnSubmit.disabled = true;
  btnSubmit.innerHTML = `
    <svg class="animate-spin -ml-1 mr-2 h-4 w-4 text-white" fill="none" viewBox="0 0 24 24">
      <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
      <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
    </svg>
    <span>กำลังประมวลผลด้วย AI...</span>
  `;

  const payload = {
    merchant: document.getElementById('input-merchant').value.trim(),
    memo: document.getElementById('input-memo').value.trim(),
    amount: parseFloat(document.getElementById('input-amount').value),
    date: document.getElementById('input-date').value.trim(),
    time: document.getElementById('input-time').value.trim()
  };

  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    if (!res.ok) {
      const errData = await res.json();
      throw new Error(errData.detail || 'Prediction failed');
    }

    const result = await res.json();
    renderPredictionResult(result);

  } catch (err) {
    alert(`การทำนายล้มเหลว: ${err.message}`);
  } finally {
    btnSubmit.disabled = false;
    btnSubmit.innerHTML = originalBtnText;
  }
}

function renderPredictionResult(res) {
  // Hide empty state and show result box
  document.getElementById('pred-empty-state').classList.add('hidden');
  const resultBox = document.getElementById('pred-result-box');
  resultBox.classList.remove('hidden');

  // Category & Confidence
  document.getElementById('pred-category-name').textContent = res.predicted_category;
  document.getElementById('pred-category-conf').textContent = `Confidence ${(res.category_confidence * 100).toFixed(1)}%`;

  // Needs vs Wants
  const nwTag = document.getElementById('pred-needs-wants-tag');
  nwTag.textContent = res.is_wants ? 'Wants (ฟุ่มเฟือย)' : 'Needs (จำเป็น)';
  nwTag.className = `text-lg font-bold ${res.is_wants ? 'text-amber-400' : 'text-emerald-400'}`;
  
  const nwReason = document.getElementById('pred-needs-wants-reason');
  nwReason.textContent = res.needs_wants_reason;
  nwReason.title = res.needs_wants_reason;

  // v1 Score & Breakdown
  document.getElementById('pred-v1-score').textContent = res.impulse_score_v1;
  const scoreBar = document.getElementById('pred-score-bar');
  scoreBar.style.width = `${res.impulse_score_v1}%`;

  document.getElementById('chip-late').textContent = `${res.score_breakdown.late_night_score} pts`;
  document.getElementById('chip-payday').textContent = `${res.score_breakdown.payday_score} pts`;
  document.getElementById('chip-wants').textContent = `${res.score_breakdown.wants_score} pts`;
  document.getElementById('chip-anomaly').textContent = `${res.score_breakdown.anomaly_score} pts`;

  // v2 ML Probability
  document.getElementById('pred-v2-prob').textContent = `${(res.impulse_probability_v2 * 100).toFixed(1)}%`;

  // Risk Badge
  const riskBadge = document.getElementById('pred-risk-badge');
  riskBadge.textContent = res.risk_level;
  if (res.is_nudge_alert) {
    riskBadge.className = 'px-2.5 py-1 text-xs font-semibold rounded-full bg-rose-500/20 text-rose-300 border border-rose-500/30';
  } else if (res.impulse_score_v1 >= 40 || res.impulse_probability_v2 >= 0.4) {
    riskBadge.className = 'px-2.5 py-1 text-xs font-semibold rounded-full bg-amber-500/20 text-amber-300 border border-amber-500/30';
  } else {
    riskBadge.className = 'px-2.5 py-1 text-xs font-semibold rounded-full bg-emerald-500/20 text-emerald-300 border border-emerald-500/30';
  }

  // Nudge Box
  const nudgeBox = document.getElementById('pred-nudge-box');
  if (res.is_nudge_alert) {
    nudgeBox.classList.remove('hidden');
  } else {
    nudgeBox.classList.add('hidden');
  }
}

function getCategoryColor(cat) {
  switch (cat.toLowerCase()) {
    case 'food': return 'bg-orange-500/10 text-orange-400 border border-orange-500/20';
    case 'shopping': return 'bg-purple-500/10 text-purple-400 border border-purple-500/20';
    case 'transport': return 'bg-blue-500/10 text-blue-400 border border-blue-500/20';
    case 'bills': return 'bg-cyan-500/10 text-cyan-400 border border-cyan-500/20';
    case 'entertainment': return 'bg-pink-500/10 text-pink-400 border border-pink-500/20';
    default: return 'bg-slate-500/10 text-slate-400 border border-slate-500/20';
  }
}

function escapeHtml(str) {
  if (!str) return '';
  return str.replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#039;');
}
