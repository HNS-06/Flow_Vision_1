# 💧 FlowVision 2.0 – Smart Water Network Intelligence

![FlowVision Dashboard](https://via.placeholder.com/1200x600?text=FlowVision+Dashboard+Mockup)

**FlowVision** is an advanced AI-powered water management system designed to detect leaks, predict consumption, and visualize network health in real-time. It seamlessly integrates a high-performance **FastAPI backend** with a responsive **Web Dashboard** and a native **Android Mobile App**.

---

## 🚀 The "Secret Sauce"

What makes FlowVision unique?

### 1. 🧠 AI-Driven Anomaly Detection (Isolation Forest)
We don't just use simple thresholds. FlowVision employs an unsupervised **Isolation Forest** model that learns "normal" flow patterns over time. It can detect subtle deviations—like a slowly growing leak—that traditional rule-based systems miss.
- **Model:** `sklearn.ensemble.IsolationForest`
- **Logic:** `ml_pipeline/leak_detection.py`

### 2. 🔮 Predictive Forecasting (Linear Regression + Feature Engineering)
FlowVision predicts water demand for the next 24 hours with high accuracy.
- **Engineered Features:** Uses lag features (t-1h, t-24h), rolling averages, and cyclical time encoding (sin/cos of hour).
- **Benefit:** Helps utilities optimize pressure and reduce energy costs.

### 3. ⚡ Real-Time WebSocket Streaming
No manual refreshing. The backend streams sensor data (flow rate, pressure, leak probability) to both the Web Dashboard and Android App instantly via **WebSockets**.
- **Latency:** < 50ms updates.

### 4. 📱 Unified Cross-Platform Experience (Capacitor)
One codebase, everywhere. The frontend is built with vanilla HTML/JS for maximum performance and wrapped with **Capacitor** to run natively on Android.
- **Web:** Accessible via browser.
- **Mobile:** Installed as a native Android APK.

---

## 🛠️ Tech Stack

- **Backend:** Python, FastAPI, Uvicorn, WebSockets.
- **Frontend:** Vanilla JS, Chart.js, Lucide Icons, CSS Variables (Dark Mode).
- **Mobile:** Capacitor, Android Studio (Gradle).
- **ML/AI:** Scikit-learn, Pandas, NumPy.
- **Data:** Synthetic data generation engine simulating realistic hydraulic behaviors.

---

## ⚡ Quick Start

### Prerequisites
- Python 3.9+
- Node.js & npm (for Mobile App build only)

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Run Everything (One-Click)
Double-click `run_flowvision.bat` or run:
```bash
run_flowvision.bat
```
This script will:
- Generate synthetic data.
- Train AI models.
- Start the server at `http://localhost:8000`.

### 3. Mobile App (Android)
To run the Android app continuously (requires Android device connected via USB):
```bash
npx cap run android
```

---

## 📂 Project Structure

```
flowvision2/
├── backend/            # FastAPI App & Routes
│   ├── app.py          # Main Server Entry
│   ├── static/         # Web Frontend (HTML/JS/CSS)
├── ml_pipeline/        # AI Models (Training & Inference)
├── android/            # Native Android Project Source
├── data/               # Datasets
├── scripts/            # Utilities
└── README.md           # This file
```

---

## 🌟 Key Features

- **Live Flow Monitoring:** Visualize L/min flow rates in real-time.
- **Leak Alerts:** Instant notifications when anomaly score > 75%.
- **Scenario Injection:** Toggle "Simulate Leak" mode to demonstrate detection capabilities.
- **Dark Mode UI:** Sleek, modern interface designed for control rooms.

---

*Built for the Future of Water Management.*
