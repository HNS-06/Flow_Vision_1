/**
 * FlowVision - Chart.js Configuration
 */

// Global Chart defaults
Chart.defaults.color = '#94a3b8';
Chart.defaults.borderColor = '#334155';
Chart.defaults.font.family = "'Inter', sans-serif";
Chart.defaults.plugins.tooltip.backgroundColor = '#1e293b';
Chart.defaults.plugins.tooltip.borderColor = '#334155';
Chart.defaults.plugins.tooltip.borderWidth = 1;
Chart.defaults.plugins.tooltip.padding = 12;
Chart.defaults.plugins.legend.labels.usePointStyle = true;

const chartInstances = {};

function initFlowChart(ctx) {
    return new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Flow Rate (L/min)',
                data: [],
                borderColor: '#3b82f6',
                backgroundColor: 'rgba(59, 130, 246, 0.1)',
                borderWidth: 2,
                tension: 0.4,
                fill: true,
                pointRadius: 0,
                pointHoverRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    grid: { display: false },
                    ticks: { maxTicksLimit: 8 }
                },
                y: {
                    beginAtZero: true,
                    grid: { borderDash: [4, 4] }
                }
            },
            plugins: {
                legend: { display: false }
            },
            animation: { duration: 0 } // Disable animation for real-time performance
        }
    });
}

function initAnomalyChart(ctx) {
    return new Chart(ctx, {
        type: 'bar',
        data: {
            labels: [],
            datasets: [{
                label: 'Anomaly Score',
                data: [],
                backgroundColor: (context) => {
                    const value = context.raw;
                    return value > 60 ? '#ef4444' : value > 30 ? '#f59e0b' : '#10b981';
                },
                borderRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { display: false },
                y: { max: 100, min: 0 }
            },
            plugins: {
                legend: { display: false }
            }
        }
    });
}

function initWardChart(ctx) {
    return new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: [],
            datasets: [{
                data: [],
                backgroundColor: [
                    '#3b82f6', '#8b5cf6', '#10b981', '#f59e0b',
                    '#ef4444', '#06b6d4', '#ec4899', '#6366f1'
                ],
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: '70%',
            plugins: {
                legend: { position: 'right' }
            }
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
                    borderColor: '#8b5cf6',
                    backgroundColor: 'rgba(139, 92, 246, 0.1)',
                    borderWidth: 2,
                    borderDash: [5, 5],
                    tension: 0.4,
                    fill: false
                },
                {
                    label: 'Upper Bound',
                    data: [],
                    borderColor: 'transparent',
                    backgroundColor: 'rgba(139, 92, 246, 0.1)',
                    fill: '+1', // Fill to lower bound
                    pointRadius: 0
                },
                {
                    label: 'Lower Bound',
                    data: [],
                    borderColor: 'transparent',
                    fill: false,
                    pointRadius: 0
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                intersect: false,
                mode: 'index',
            }
        }
    });
}

// Update functions
function updateRealTimeCharts(flowChart, anomalyChart, data) {
    const timestamp = new Date(data.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });

    // Add new data
    flowChart.data.labels.push(timestamp);
    flowChart.data.datasets[0].data.push(data.flow_rate);

    anomalyChart.data.labels.push(timestamp);
    anomalyChart.data.datasets[0].data.push(data.leak_probability);

    // Remove old data (keep last 30 points)
    if (flowChart.data.labels.length > 30) {
        flowChart.data.labels.shift();
        flowChart.data.datasets[0].data.shift();

        anomalyChart.data.labels.shift();
        anomalyChart.data.datasets[0].data.shift();
    }

    flowChart.update('none'); // 'none' mode for performance
    anomalyChart.update('none');
}
