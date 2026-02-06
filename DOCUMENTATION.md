# FlowVision AI Documentation

This document provides a technical overview of the algorithms, datasets, and models used in the FlowVision system.

## 1. Datasets

The system uses synthetic data generated to replicate realistic municipal water network behavior containing 5000+ hourly records.

-   **Source**: generated via `scripts/generate_sample_data.py`
-   **Structure**:
    -   **Water Flow (`water_flow.csv`)**: Hourly flow rate (L/min), pressure (PSI), and temperature.
    -   **Consumption (`consumption_daily.csv`)**: Daily aggregated consumption per ward.
    -   **Leak Labels (`leak_labels.csv`)**: Ground truth generated leak events for evaluation.
    -   **Ward Usage (`ward_usage.csv`)**: Demographic and usage profiles for 12 municipal wards.

## 2. Machine Learning Models

### A. Leak Detection (Anomaly Detection)
We use an **Ensemble Approach** combining statistical and Machine Learning methods to detect leaks with high precision.

1.  **Isolation Forest (Unsupervised ML)**:
    -   **Library**: `sklearn.ensemble.IsolationForest`
    -   **Role**: Detects complex, non-linear anomalies in high-dimensional space (flow, pressure, time).
    -   **Features**: Flow rate, rolling mean/std (3h, 6h, 12h), differential flow, pressure.
    -   **Contamination**: Set to 5% (expected anomaly rate).

2.  **Statistical Z-Score**:
    -   Detects extreme deviations from the global mean (> 3 standard deviations).
    -   Best for identifying sudden massive bursts.

3.  **Rolling Deviation**:
    -   Compares current flow against a moving average (24-hour window).
    -   Detects gradual drifts or sustained leaks that don't trigger global thresholds.

4.  **Night Flow Analysis**:
    -   Monitors flow between 00:00 - 05:00.
    -   High minimum night flow is a strong indicator of background leakage.

**Final Scoring**: The system calculates a weighted probability (0-100%) based on votes from all 4 methods.

### B. Consumption Forecasting
We use a hybrid forecasting engine to predict future water demand.

1.  **Linear Regression**:
    -   **Library**: `sklearn.linear_model.LinearRegression`
    -   **Role**: Captures the base trend and daily seasonality.
    -   **Features**: Hour of day (one-hot encoded), day of week, is_weekend, lag features (t-1, t-24).

2.  **ARIMA (Time Series)**:
    -   **Library**: `statsmodels.tsa.arima.model.ARIMA`
    -   **Role**: Captures temporal dependencies and residuals left by the linear model.
    -   **Order**: (1, 1, 1) - Auto-Regressive, Integrated, Moving Average.

## 3. Real-time Architecture

-   **Backend**: FastAPI (Python)
-   **Simulation**: A background service (`simulation_service.py`) streams real-time sensor data via **WebSockets**.
-   **Frontend**: Vanilla JavaScript Dashboard updates charts `50ms` after receiving data packets.
