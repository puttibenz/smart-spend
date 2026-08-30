/**
 * SmartSpend AI - Clean Modern Dashboard (Stripe SaaS Style + GitHub Heatmap)
 * Handles API data fetching, Chart.js visualizations, GitHub-style 7x24 Matrix rendering,
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

    // 3. Fetch & render GitHub-style heatmap
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
        backgroundColor: ['#10b981', '#f43f5e'],
        borderColor: '#ffffff',
        borderWidth: 4,
        hoverOffset: 6
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: '74%',
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: '#0f172a',
          titleColor: '#ffffff',
          bodyColor: '#e2e8f0',
          padding: 10,
          cornerRadius: 8,
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
          borderColor: '#4f46e5',
          backgroundColor: 'rgba(79, 70, 229, 0.08)',
          fill: true,
          tension: 0.35,
          borderWidth: 2.5,
          pointRadius: 3.5,
          pointBackgroundColor: '#4f46e5'
        },
        {
          label: 'ยอด Impulse Buying',
          data: impulseAmounts,
          borderColor: '#f43f5e',
          backgroundColor: 'rgba(244, 63, 94, 0.08)',
          fill: true,
          tension: 0.35,
          borderWidth: 2.5,
          pointRadius: 4,
          pointBackgroundColor: '#f43f5e'
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          grid: { color: '#f1f5f9' },
          ticks: { color: '#64748b', font: { size: 11, family: 'Inter' } }
        },
        y: {
          grid: { color: '#f1f5f9' },
          ticks: {
            color: '#64748b',
            font: { size: 11, family: 'Inter' },
            callback: value => `฿${(value / 1000).toFixed(0)}k`
          }
        }
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: '#0f172a',
          titleColor: '#ffffff',
          bodyColor: '#e2e8f0',
          padding: 10,
          cornerRadius: 8,
          callbacks: {
            label: context => ` ${context.dataset.label}: ฿${context.raw.toLocaleString()}`
          }
        }
      }
    }
  });
}

/**
 * Fetches and renders the 7 Days x 24 Hours Heatmap in GitHub Contribution Style.
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

  // GitHub 5-level green color palette
  const GH_COLORS = {
    0: '#ebedf0', // No transactions
    1: '#9be9a8', // Level 1 (Light green)
    2: '#40c463', // Level 2 (Medium green)
    3: '#30a14e', // Level 3 (Deep green)
    4: '#216e39', // Level 4 (Dark rich green)
  };

  let html = `
    <div class="grid grid-cols-[80px_repeat(24,minmax(0,1fr))] gap-1.5 text-center text-[10px]">
      <!-- Header: Hours 00 to 23 -->
      <div class="py-1 text-slate-400 font-semibold">วัน \ ชม.</div>
  `;

  for (let h = 0; h < 24; h++) {
    const isLate = (h >= 23 || h <= 2);
    html += `<div class="py-1 font-mono font-semibold ${isLate ? 'text-rose-500 font-bold' : 'text-slate-400'}">${String(h).padStart(2, '0')}</div>`;
  }

  // Rows for each Day
  for (let d = 0; d < 7; d++) {
    html += `<div class="text-left font-semibold text-slate-600 py-1.5 truncate pr-1">${days[d].split(' ')[0]}</div>`;
    for (let h = 0; h < 24; h++) {
      const cell = matrix[d][h];
      const amt = cell.total_amount;
      const count = cell.transaction_count;
      const imp = cell.impulse_count;
      const isLate = cell.is_late_night;

      // GitHub color mapping based on amount intensity
      let bgColor = GH_COLORS[0];
      let borderStyle = 'border: 1px solid #e2e8f0;';
      if (amt > 0) {
        const ratio = amt / maxAmt;
        borderStyle = 'border: 1px solid rgba(0, 0, 0, 0.05);';
        if (ratio > 0.6) {
          bgColor = GH_COLORS[4];
        } else if (ratio > 0.3) {
          bgColor = GH_COLORS[3];
        } else if (ratio > 0.1) {
          bgColor = GH_COLORS[2];
        } else {
          bgColor = GH_COLORS[1];
        }
      }

      // Indicator for impulse transactions during late night
      const hasImpulseIndicator = (imp > 0 && isLate);

      html += `
        <div class="gh-cell relative rounded-[3px] h-7 flex items-center justify-center cursor-pointer shadow-xs"
             style="background-color: ${bgColor}; ${borderStyle}"
             title="${days[d]} ${String(h).padStart(2, '0')}:00\n• ยอดใช้จ่ายรวม: ฿${amt.toLocaleString()}\n• จำนวน: ${count} รายการ\n• Impulse Risk: ${imp} รายการ">
          ${hasImpulseIndicator ? `<span class="w-1.5 h-1.5 rounded-full bg-rose-500 ring-2 ring-white animate-pulse"></span>` : ''}
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
        <td colspan="7" class="px-4 py-8 text-center text-slate-400">
          ไม่พบรายการที่ตรงกับเงื่อนไขการค้นหา
        </td>
      </tr>
    `;
  } else {
    tbody.innerHTML = data.items.map(tx => {
      const categoryBadge = getCategoryBadge(tx.category);
      const isNudge = tx.is_nudge_alert;

      return `
        <tr class="hover:bg-slate-50/80 transition ${isNudge ? 'bg-rose-50/40' : ''}">
          <td class="px-4 py-3 font-mono text-slate-500 whitespace-nowrap">
            <div class="font-medium text-slate-700">${tx.date}</div>
            <div class="text-[10px] text-slate-400">${tx.time}</div>
          </td>
          <td class="px-4 py-3">
            <div class="font-bold text-slate-900">${escapeHtml(tx.merchant)}</div>
            ${tx.memo ? `<div class="text-[11px] text-slate-500 truncate max-w-xs">${escapeHtml(tx.memo)}</div>` : ''}
          </td>
          <td class="px-4 py-3 text-right font-mono font-bold text-slate-900 whitespace-nowrap">
            ฿${tx.amount.toLocaleString('en-US', { minimumFractionDigits: 2 })}
          </td>
          <td class="px-4 py-3">
            ${categoryBadge}
          </td>
          <td class="px-4 py-3 text-center">
            <span class="px-2.5 py-0.5 rounded-full text-[10px] font-bold ${tx.is_wants ? 'bg-rose-50 text-rose-700 border border-rose-200' : 'bg-emerald-50 text-emerald-700 border border-emerald-200'}">
              ${tx.is_wants ? 'Wants' : 'Needs'}
            </span>
          </td>
          <td class="px-4 py-3 text-center">
            <span class="font-mono font-bold ${tx.impulse_score >= appState.nudgeThreshold ? 'text-rose-600 font-extrabold' : (tx.impulse_score >= 40 ? 'text-amber-600' : 'text-slate-600')}">
              ${tx.impulse_score}
            </span>
          </td>
          <td class="px-4 py-3 text-center whitespace-nowrap">
            ${isNudge ? 
              `<span class="px-2 py-0.5 rounded-full bg-rose-100 text-rose-700 border border-rose-200 text-[10px] font-bold inline-flex items-center gap-1"><span class="w-1.5 h-1.5 rounded-full bg-rose-500 animate-ping"></span>🚨 Nudge</span>` : 
              `<span class="text-slate-300">-</span>`
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
  nwTag.className = `text-lg font-extrabold ${res.is_wants ? 'text-rose-600' : 'text-emerald-600'}`;
  
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
    riskBadge.className = 'px-3 py-1 text-xs font-bold rounded-full bg-rose-100 text-rose-700 border border-rose-200';
  } else if (res.impulse_score_v1 >= 40 || res.impulse_probability_v2 >= 0.4) {
    riskBadge.className = 'px-3 py-1 text-xs font-bold rounded-full bg-amber-100 text-amber-700 border border-amber-200';
  } else {
    riskBadge.className = 'px-3 py-1 text-xs font-bold rounded-full bg-emerald-100 text-emerald-700 border border-emerald-200';
  }

  // Nudge Box
  const nudgeBox = document.getElementById('pred-nudge-box');
  if (res.is_nudge_alert) {
    nudgeBox.classList.remove('hidden');
  } else {
    nudgeBox.classList.add('hidden');
  }
}

function getCategoryBadge(cat) {
  switch (cat.toLowerCase()) {
    case 'food': 
      return `<span class="px-2.5 py-0.5 rounded-full text-[10px] font-semibold bg-emerald-50 text-emerald-700 border border-emerald-200/80">food</span>`;
    case 'shopping': 
      return `<span class="px-2.5 py-0.5 rounded-full text-[10px] font-semibold bg-purple-50 text-purple-700 border border-purple-200/80">shopping</span>`;
    case 'transport': 
      return `<span class="px-2.5 py-0.5 rounded-full text-[10px] font-semibold bg-blue-50 text-blue-700 border border-blue-200/80">transport</span>`;
    case 'bills': 
      return `<span class="px-2.5 py-0.5 rounded-full text-[10px] font-semibold bg-amber-50 text-amber-700 border border-amber-200/80">bills</span>`;
    case 'entertainment': 
      return `<span class="px-2.5 py-0.5 rounded-full text-[10px] font-semibold bg-pink-50 text-pink-700 border border-pink-200/80">entertainment</span>`;
    default: 
      return `<span class="px-2.5 py-0.5 rounded-full text-[10px] font-semibold bg-slate-100 text-slate-700 border border-slate-200">other</span>`;
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
