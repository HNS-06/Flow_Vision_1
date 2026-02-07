/**
 * FlowVision - Smart Water Network Dashboard
 * Complete implementation with all views and real-time data
 */

// Configuration
// Configuration
const API_BASE_URL = `http://${window.location.hostname}:8000`;

// State
let ws = null;
let currentWard = 1;
let selectedView = 'dashboard';
let simulationActive = false;
let charts = {};


// DOM Elements
const elements = {
    metrics: {
        flow: document.getElementById('metric-flow'),
        pressure: document.getElementById('metric-pressure'),
        leakProb: document.getElementById('metric-leak-prob'),
        consumption: document.getElementById('metric-consumption')
    }
};

// Initialize
async function initApp() {
    console.log("FlowVision Initializing...");

    // Setup Ward Selector
    setupWardSelector();

    // Initialize ALL Charts
    if (document.getElementById('flowChart')) {
        charts.flow = initFlowChart(document.getElementById('flowChart').getContext('2d'));
    }

    if (document.getElementById('anomalyChart')) {
        charts.anomaly = initAnomalyChart(document.getElementById('anomalyChart').getContext('2d'));
    }

    if (document.getElementById('wardChart')) {
        charts.ward = initWardChart(document.getElementById('wardChart').getContext('2d'));
    }

    if (document.getElementById('forecastChart')) {
        charts.forecast = initForecastChart(document.getElementById('forecastChart').getContext('2d'));
    }

    if (document.getElementById('simulationChart')) {
        charts.simulation = initFlowChart(document.getElementById('simulationChart').getContext('2d'));
    }

    // Load initial data
    await loadWardData();
    await loadForecastData();
    await loadInsightsData();

    // Connect WebSocket for real-time updates
    connectWebSocket();

    // Event Listeners
    const leakBtn = document.getElementById('scenario-leak');
    const normalBtn = document.getElementById('scenario-normal');
    if (leakBtn) leakBtn.addEventListener('click', () => toggleScenario(true));
    if (normalBtn) normalBtn.addEventListener('click', () => toggleScenario(false));

    // Setup Navigation
    setupNavigation();

    // Start Clock
    setInterval(updateClock, 1000);
    updateClock();

    console.log("FlowVision Ready!");
}

// Load Ward-wise Consumption Data
async function loadWardData() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/data/wards`);
        const data = await response.json();
        if (data.success && charts.ward) {
            const labels = data.data.map(w => w.ward_name);
            const values = data.data.map(w => w.avg_daily_consumption_m3);
            charts.ward.data.labels = labels;
            charts.ward.data.datasets[0].data = values;
            charts.ward.update();
            console.log("Ward data loaded:", values);
        }
    } catch (error) {
        console.error("Failed to load ward data:", error);
    }
}

// Load Forecast Data
async function loadForecastData() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/ml/forecast`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ steps: 24, ward_id: currentWard })
        });
        const data = await response.json();
        if (data.success && charts.forecast && data.forecast) {
            const labels = data.forecast.map((_, i) => `H${i + 1}`);
            const predictions = data.forecast.map(f => f.prediction || f.forecast || 0);
            const upper = data.forecast.map(f => f.upper_bound || f.prediction * 1.1 || 0);
            const lower = data.forecast.map(f => f.lower_bound || f.prediction * 0.9 || 0);

            charts.forecast.data.labels = labels;
            charts.forecast.data.datasets[0].data = predictions;
            charts.forecast.data.datasets[1].data = upper;
            charts.forecast.data.datasets[2].data = lower;
            charts.forecast.update();
            console.log("Forecast data loaded");
        }
    } catch (error) {
        console.error("Failed to load forecast:", error);
    }
}

// Load AI Insights
async function loadInsightsData() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/ml/insights`);
        const data = await response.json();
        if (data.success && data.insights) {
            renderInsights(data.insights);
        }
    } catch (error) {
        console.error("Failed to load insights:", error);
        // Render default insights
        renderInsights({
            consumption_trend: '+12.5%',
            leak_risk: 'Low',
            peak_hour: '8:00 AM',
            efficiency: '94.2%'
        });
    }
}

// Render AI Insights View
function renderInsights(insights) {
    const container = document.getElementById('insights-container');
    if (!container) return;

    container.innerHTML = `
        <div class="insights-grid">
            <div class="insight-card" style="background: rgba(255,255,255,0.05); border-radius: 12px; padding: 1.5rem; border: 1px solid rgba(255,255,255,0.1);">
                <div class="insight-icon" style="width: 50px; height: 50px; border-radius: 10px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); display: flex; align-items: center; justify-content: center; margin-bottom: 1rem;">
                    <i data-lucide="trending-up" style="color: white;"></i>
                </div>
                <h3 style="font-size: 0.9rem; color: #94a3b8; margin-bottom: 0.5rem;">Consumption Trend</h3>
                <p style="font-size: 1.8rem; font-weight: 600; color: #00f2ea; margin-bottom: 0.25rem;">${insights.consumption_trend || '+12.5%'}</p>
                <p style="font-size: 0.8rem; color: #64748b;">Compared to last week</p>
            </div>
            
            <div class="insight-card" style="background: rgba(255,255,255,0.05); border-radius: 12px; padding: 1.5rem; border: 1px solid rgba(255,255,255,0.1);">
                <div class="insight-icon" style="width: 50px; height: 50px; border-radius: 10px; background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); display: flex; align-items: center; justify-content: center; margin-bottom: 1rem;">
                    <i data-lucide="alert-triangle" style="color: white;"></i>
                </div>
                <h3 style="font-size: 0.9rem; color: #94a3b8; margin-bottom: 0.5rem;">Leak Risk Score</h3>
                <p style="font-size: 1.8rem; font-weight: 600; color: #00f2ea; margin-bottom: 0.25rem;">${insights.leak_risk || 'Low'}</p>
                <p style="font-size: 0.8rem; color: #64748b;">AI-powered prediction</p>
            </div>
            
            <div class="insight-card" style="background: rgba(255,255,255,0.05); border-radius: 12px; padding: 1.5rem; border: 1px solid rgba(255,255,255,0.1);">
                <div class="insight-icon" style="width: 50px; height: 50px; border-radius: 10px; background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); display: flex; align-items: center; justify-content: center; margin-bottom: 1rem;">
                    <i data-lucide="droplets" style="color: white;"></i>
                </div>
                <h3 style="font-size: 0.9rem; color: #94a3b8; margin-bottom: 0.5rem;">Peak Usage Hour</h3>
                <p style="font-size: 1.8rem; font-weight: 600; color: #00f2ea; margin-bottom: 0.25rem;">${insights.peak_hour || '8:00 AM'}</p>
                <p style="font-size: 0.8rem; color: #64748b;">Highest consumption period</p>
            </div>
            
            <div class="insight-card" style="background: rgba(255,255,255,0.05); border-radius: 12px; padding: 1.5rem; border: 1px solid rgba(255,255,255,0.1);">
                <div class="insight-icon" style="width: 50px; height: 50px; border-radius: 10px; background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); display: flex; align-items: center; justify-content: center; margin-bottom: 1rem;">
                    <i data-lucide="activity" style="color: white;"></i>
                </div>
                <h3 style="font-size: 0.9rem; color: #94a3b8; margin-bottom: 0.5rem;">System Efficiency</h3>
                <p style="font-size: 1.8rem; font-weight: 600; color: #00f2ea; margin-bottom: 0.25rem;">${insights.efficiency || '94.2%'}</p>
                <p style="font-size: 0.8rem; color: #64748b;">Overall performance</p>
            </div>
        </div>
        
        <div class="recommendations-section" style="background: rgba(255,255,255,0.05); border-radius: 12px; padding: 1.5rem; border: 1px solid rgba(255,255,255,0.1);">
            <h2 style="font-size: 1.2rem; margin-bottom: 1rem; color: #e2e8f0;">AI Recommendations</h2>
            <div class="recommendation-list" style="display: flex; flex-direction: column; gap: 1rem;">
                <div style="display: flex; align-items: start; gap: 0.75rem;">
                    <i data-lucide="check-circle" style="color: #00f2ea; flex-shrink: 0; margin-top: 2px;"></i>
                    <span style="color: #cbd5e1;">Ward 2 shows optimal consumption patterns - use as benchmark</span>
                </div>
                <div style="display: flex; align-items: start; gap: 0.75rem;">
                    <i data-lucide="alert-circle" style="color: #ff0055; flex-shrink: 0; margin-top: 2px;"></i>
                    <span style="color: #cbd5e1;">Ward 3 pressure fluctuations detected - schedule maintenance check</span>
                </div>
                <div style="display: flex; align-items: start; gap: 0.75rem;">
                    <i data-lucide="info" style="color: #3b82f6; flex-shrink: 0; margin-top: 2px;"></i>
                    <span style="color: #cbd5e1;">Peak hours: 7-9 AM and 6-8 PM - consider load balancing</span>
                </div>
            </div>
        </div>
    `;

    if (window.lucide) lucide.createIcons();
}

// Setup Ward Selector
function setupWardSelector() {
    let select = document.getElementById('ward-select');
    if (!select) {
        const header = document.querySelector('.header-actions');
        if (header) {
            select = document.createElement('select');
            select.id = 'ward-select';
            select.className = 'ward-selector';
            select.style.cssText = 'padding: 0.5rem 1rem; border-radius: 8px; background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.2); color: white; margin-right: 1rem;';
            [1, 2, 3, 4].forEach(i => {
                const opt = document.createElement('option');
                opt.value = i;
                opt.innerText = `Ward ${i}`;
                select.appendChild(opt);
            });
            header.insertBefore(select, header.firstChild);
        }
    }

    if (select) {
        select.addEventListener('change', (e) => {
            currentWard = parseInt(e.target.value);
            console.log(`Switched to Ward ${currentWard}`);

            // Update all headings with the new ward number
            if (headingTitle) {
                // Update main heading based on current view
                const activeView = document.querySelector('.page-view.active');
                if (activeView) {
                    const viewId = activeView.id;
                    if (viewId === 'view-analytics') {
                        headingTitle.textContent = `Advanced Analytics - Ward ${currentWard}`;
                    } else if (viewId === 'view-insights') {
                        headingTitle.textContent = `AI Insights - Ward ${currentWard}`;
                    }
                }
            }

            // Reload Data
            loadForecastData();
            loadInsightsData();
        });
    }
}

// Simulation Controls
function updateSimulationStatus(status) {
    const statusEl = document.getElementById('sim-status');
    if (statusEl) {
        statusEl.textContent = status;
        statusEl.style.color = status === 'RUNNING' ? '#00f2ea' : (status === 'STOPPED' ? '#ff0055' : '#94a3b8');
    }
}

async function startSimulation() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/simulation/start`, { method: 'POST' });
        if (response.ok) {
            simulationActive = true;
            updateSimulationStatus('RUNNING');
            console.log('Simulation started');
        }
    } catch (error) {
        console.error('Failed to start simulation:', error);
    }
}

async function stopSimulation() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/simulation/stop`, { method: 'POST' });
        if (response.ok) {
            simulationActive = false;
            updateSimulationStatus('STOPPED');
            console.log('Simulation stopped');
        }
    } catch (error) {
        console.error('Failed to stop simulation:', error);
    }
}

// WebSocket Connection
function connectWebSocket() {
    // Connect to backend IP directly
    ws = new WebSocket(`ws://${window.location.hostname}:8000/api/simulation/ws`);

    ws.onopen = () => {
        console.log("WebSocket connected");
        simulationActive = true; // Auto-start to show data immediately
        updateSimulationStatus('RUNNING');
        // Start simulation automatically
        fetch(`${API_BASE_URL}/api/simulation/start`, { method: 'POST' })
            .then(() => console.log('Simulation auto-started'))
            .catch(err => console.error('Failed to auto-start:', err));
    };

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.wards) {
            processMultiWardData(data);
        }
    };

    ws.onclose = () => {
        console.log("WebSocket disconnected, reconnecting...");
        setTimeout(connectWebSocket, 3000);
    };

    ws.onerror = (error) => {
        console.error("WebSocket error:", error);
    };
}

// Process Multi-Ward Data
function processMultiWardData(rootData) {
    const wardKey = `ward_${currentWard}`;
    const wardData = rootData.wards[wardKey];

    if (!wardData) return;

    // 1. Update Metrics
    updateMetrics(wardData);

    // 2. Render Smart Grid
    renderSmartGrid(wardData);

    // 3. Handle Alerts
    if (wardData.alerts && wardData.alerts.length > 0) {
        showAlerts(wardData.alerts);
    }

    // 4. Update Charts
    const totalFlow = wardData.pipes.reduce((sum, p) => sum + p.flow_rate, 0);
    const realTimeData = {
        flow_rate: totalFlow,
        leak_probability: wardData.leak_probability
    };

    updateRealTimeCharts(charts.flow, charts.anomaly, realTimeData);

    // Also update simulation chart if it exists
    if (charts.simulation) {
        updateRealTimeCharts(charts.simulation, null, realTimeData);
    }
}

// Render Smart Grid
function renderSmartGrid(wardData) {
    let container = document.getElementById('smart-grid-visual');
    if (!container) {
        const parent = document.querySelector('.main-content') || document.querySelector('#view-dashboard');
        if (!parent) return;
        container = document.createElement('div');
        container.id = 'smart-grid-visual';
        container.style.cssText = 'margin-bottom: 2rem;';
        parent.insertBefore(container, parent.firstChild);
    }

    const pipesHtml = wardData.pipes.map(pair => {
        const isBackupActive = pair.active_pipe === pair.backup_pipe;
        const isLeaking = pair.is_leaking;

        return `
            <div style="background: rgba(255,255,255,0.05); border-radius: 12px; padding: 1.5rem; border: 1px solid rgba(255,255,255,0.1); margin-bottom: 1rem;">
                <h4 style="color:#94a3b8; margin-bottom: 1rem;">Pipe Pair ${pair.pair_id}</h4>
                
                <div style="display: flex; gap: 1rem; margin-bottom: 0.5rem;">
                    <div style="flex: 1; padding: 1rem; border-radius: 8px; background: ${isLeaking ? 'rgba(255,0,85,0.2)' : 'rgba(0,242,234,0.1)'}; border: 2px solid ${isLeaking ? '#ff0055' : '#00f2ea'}; opacity: ${isBackupActive ? 0.5 : 1};">
                        <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
                            <span style="font-size: 1.5rem;">🟢</span>
                            <strong style="color: white;">Pipe-${pair.primary_pipe} (Primary)</strong>
                        </div>
                        <div style="color: #94a3b8; font-size: 0.9rem;">Flow: ${isBackupActive ? 0 : pair.flow_rate.toFixed(1)} L/min</div>
                        <div style="color: ${isLeaking ? '#ff0055' : '#00f2ea'}; font-weight: 600; margin-top: 0.5rem;">
                            ${isLeaking ? '⚠️ LEAK DETECTED' : (isBackupActive ? 'CLOSED' : '✓ ACTIVE')}
                        </div>
                    </div>
                    
                    <div style="flex: 1; padding: 1rem; border-radius: 8px; background: rgba(59,130,246,0.1); border: 2px solid ${isBackupActive ? '#3b82f6' : '#64748b'}; opacity: ${!isBackupActive ? 0.5 : 1};">
                        <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
                            <span style="font-size: 1.5rem;">🔵</span>
                            <strong style="color: white;">Pipe-${pair.backup_pipe} (Backup)</strong>
                        </div>
                        <div style="color: #94a3b8; font-size: 0.9rem;">Flow: ${isBackupActive ? pair.flow_rate.toFixed(1) : 0} L/min</div>
                        <div style="color: ${isBackupActive ? '#3b82f6' : '#64748b'}; font-weight: 600; margin-top: 0.5rem;">
                            ${isBackupActive ? '✓ ACTIVE' : 'STANDBY'}
                        </div>
                    </div>
                </div>
            </div>
        `;
    }).join('');

    container.innerHTML = `
        <div style="background: rgba(255,255,255,0.05); border-radius: 12px; padding: 1.5rem; border: 1px solid rgba(255,255,255,0.1);">
            <div style="font-size: 1.2rem; font-weight: 600; margin-bottom: 1.5rem; color: #00f2ea;">🏢 MAIN TANK - WARD ${currentWard}</div>
            ${pipesHtml}
        </div>
    `;
}

// Show Alerts
function showAlerts(alerts) {
    const container = document.getElementById('alerts-list');
    if (!container) return;

    alerts.forEach(msg => {
        const div = document.createElement('div');
        div.style.cssText = 'padding: 0.75rem; background: rgba(255,0,85,0.1); border-left: 3px solid #ff0055; border-radius: 4px; margin-bottom: 0.5rem; display: flex; align-items: center; gap: 0.75rem; animation: slideIn 0.3s ease;';
        div.innerHTML = `
            <i data-lucide="alert-circle" style="color: #ff0055; flex-shrink: 0;"></i>
            <span style="color: #e2e8f0; flex: 1;">${msg}</span>
            <span style="font-size: 0.7rem; color: #64748b;">Just now</span>
        `;
        container.insertBefore(div, container.firstChild);

        while (container.children.length > 10) {
            container.removeChild(container.lastChild);
        }
    });

    if (window.lucide) lucide.createIcons();
}

// Update Metrics
function updateMetrics(wardData) {
    if (!wardData || !wardData.pipes) return;

    const totalFlow = wardData.pipes.reduce((sum, p) => sum + p.flow_rate, 0);
    const avgPressure = wardData.pipes.reduce((sum, p) => sum + p.pressure, 0) / wardData.pipes.length;
    const leakProb = wardData.leak_probability || 0;

    const flowEl = document.getElementById('metric-flow');
    const pressureEl = document.getElementById('metric-pressure');
    const leakProbEl = document.getElementById('metric-leak-prob');
    const consumptionEl = document.getElementById('metric-consumption');

    if (flowEl) flowEl.textContent = totalFlow.toFixed(1);
    if (pressureEl) pressureEl.textContent = avgPressure.toFixed(1);
    if (leakProbEl) {
        leakProbEl.textContent = leakProb.toFixed(1);
        const progressBar = document.getElementById('leak-progress');
        if (progressBar) {
            progressBar.style.width = `${leakProb}%`;
            progressBar.style.backgroundColor = leakProb > 50 ? '#ff0055' : '#00f2ea';
        }
    }
    if (consumptionEl) {
        const dailyEstimate = (totalFlow * 60 * 24) / 1000;
        consumptionEl.textContent = dailyEstimate.toFixed(1);
    }
}

// Toggle Scenario
async function toggleScenario(isLeak) {
    try {
        await fetch(`${API_BASE_URL}/api/simulation/scenario`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ leak: isLeak })
        });

        const leakBtn = document.getElementById('scenario-leak');
        const normalBtn = document.getElementById('scenario-normal');
        if (leakBtn) leakBtn.classList.toggle('active', isLeak);
        if (normalBtn) normalBtn.classList.toggle('active', !isLeak);

        console.log(`Scenario: ${isLeak ? 'LEAK' : 'NORMAL'}`);
    } catch (error) {
        console.error('Failed to toggle scenario:', error);
    }
}

// Chart Initialization Functions
function initFlowChart(ctx) {
    return new Chart(ctx, {
        type: 'line',
        data: {
            labels: Array(20).fill(''),
            datasets: [{
                label: 'Flow Rate (L/min)',
                data: Array(20).fill(0),
                borderColor: '#00f2ea',
                backgroundColor: 'rgba(0, 242, 234, 0.1)',
                tension: 0.4,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: { grid: { color: 'rgba(255, 255, 255, 0.1)' }, ticks: { color: '#94a3b8' } },
                x: { grid: { display: false }, ticks: { display: false } }
            },
            plugins: { legend: { display: false } }
        }
    });
}

function initAnomalyChart(ctx) {
    return new Chart(ctx, {
        type: 'line',
        data: {
            labels: Array(20).fill(''),
            datasets: [{
                label: 'Anomaly Score',
                data: Array(20).fill(0),
                borderColor: '#ff0055',
                borderDash: [5, 5],
                tension: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: { max: 100, min: 0, grid: { color: 'rgba(255, 255, 255, 0.1)' }, ticks: { color: '#94a3b8' } },
                x: { display: false }
            },
            plugins: { legend: { display: false } }
        }
    });
}

function initWardChart(ctx) {
    return new Chart(ctx, {
        type: 'bar',
        data: {
            labels: [],
            datasets: [{
                label: 'Avg Consumption (m³)',
                data: [],
                backgroundColor: '#3b82f6',
                borderRadius: 5
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: { grid: { color: 'rgba(255, 255, 255, 0.1)' }, ticks: { color: '#94a3b8' } },
                x: { grid: { display: false }, ticks: { color: '#94a3b8' } }
            },
            plugins: { legend: { labels: { color: '#94a3b8' } } }
        }
    });
}

function initForecastChart(ctx) {
    return new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Forecast',
                    data: [],
                    borderColor: '#00f2ea',
                    backgroundColor: 'rgba(0, 242, 234, 0.1)',
                    tension: 0.4,
                    fill: true
                },
                {
                    label: 'Upper Bound',
                    data: [],
                    borderColor: 'rgba(0, 242, 234, 0.3)',
                    borderDash: [5, 5],
                    fill: false,
                    pointRadius: 0
                },
                {
                    label: 'Lower Bound',
                    data: [],
                    borderColor: 'rgba(0, 242, 234, 0.3)',
                    borderDash: [5, 5],
                    fill: '-1',
                    backgroundColor: 'rgba(0, 242, 234, 0.05)',
                    pointRadius: 0
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: { grid: { color: 'rgba(255, 255, 255, 0.1)' }, ticks: { color: '#94a3b8' } },
                x: { grid: { display: false }, ticks: { color: '#94a3b8' } }
            },
            plugins: { legend: { labels: { color: '#94a3b8' } } }
        }
    });
}

function updateRealTimeCharts(chart, anomalyChart, data) {
    if (!chart) return;

    const labels = chart.data.labels;
    const values = chart.data.datasets[0].data;

    labels.push('');
    labels.shift();

    values.push(data.flow_rate);
    values.shift();

    chart.update('none');

    if (anomalyChart && data.leak_probability !== undefined) {
        const aValues = anomalyChart.data.datasets[0].data;
        aValues.push(data.leak_probability);
        aValues.shift();
        anomalyChart.update('none');
    }
}

// Navigation
function setupNavigation() {
    const navItems = document.querySelectorAll('.nav-item');
    navItems.forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            const target = item.dataset.page;
            updateDashboardView(target);
        });
    });
}

function updateDashboardView(targetPage) {
    if (targetPage) selectedView = targetPage;

    document.querySelectorAll('.nav-item').forEach(nav => {
        nav.classList.toggle('active', nav.dataset.page === selectedView);
    });

    document.querySelectorAll('.page-view').forEach(view => {
        view.classList.remove('active');
        if (view.id === `view-${selectedView}`) {
            view.classList.add('active');
        }
    });

    const pageTitle = document.getElementById('page-title');
    if (pageTitle) {
        let title = 'Real-time Overview';
        if (selectedView === 'insights') title = 'AI Insights';
        if (selectedView === 'simulation') title = 'System Simulation';
        if (selectedView === 'analytics') title = 'Advanced Analytics';
        title += ` - Ward ${currentWard}`;
        pageTitle.textContent = title;
    }

    Object.values(charts).forEach(c => c && c.resize && c.resize());
}

function updateClock() {
    const el = document.getElementById('current-time');
    if (el) el.textContent = new Date().toLocaleTimeString();
}

// Start
document.addEventListener('DOMContentLoaded', initApp);
