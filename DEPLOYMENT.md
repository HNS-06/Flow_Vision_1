# FlowVision Deployment Guide

This guide describes how to deploy the FlowVision Smart Water Management System. The application is a self-contained Python FastAPI application that serves the frontend static files.

## Option 1: Deploy with Docker (Recommended)

Docker is the easiest way to deploy because it bundles all dependencies, generates the necessary data, and trains the models during the build process.

### 1. Build the Docker Image
Run this command in the root of the project:

```bash
docker build -t flowvision .
```

*Note: The build process includes generating synthetic data and training models, so it may take a few minutes.*

### 2. Run the Container

```bash
docker run -p 8000:8000 flowvision
```

Access the dashboard at `http://localhost:8000`.

---

## Option 2: Cloud Deployment (Render, Railway, Heroku)

This project is ready for cloud deployment using the included `Dockerfile` or `Procfile`.

### Deploying to Render.com (Free Tier available)

1.  Push your code to a GitHub/GitLab repository.
2.  Sign up for [Render](https://render.com).
3.  Click **New +** -> **Web Service**.
4.  Connect your repository.
5.  Render will automatically detect the `Dockerfile`.
6.  Click **Create Web Service**.

*Render will build the image (including data generation) and deploy it. It usually takes 5-10 minutes for the first build.*

### Deploying to Railway.app

1.  Push code to GitHub.
2.  Login to Railway.
3.  Create a **New Project** -> **Deploy from GitHub repo**.
4.  Railway will detect the `Dockerfile` and build it.

---

## Option 3: Manual Deployment (VPS / server)

If you are deploying to a standard Linux server (Ubuntu/Debian):

1.  **Install Python 3.9+**:
    ```bash
    sudo apt update
    sudo apt install python3 python3-pip
    ```

2.  **Clone the contents** to `/opt/flowvision` (or your preferred dir).

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    pip install gunicorn
    ```

4.  **Generate Data & Train Models**:
    ```bash
    python scripts/generate_sample_data.py
    python ml_pipeline/data_preprocessing.py
    python ml_pipeline/leak_detection.py
    python ml_pipeline/consumption_forecast.py
    ```

5.  **Run with Gunicorn (Production Server)**:
    ```bash
    gunicorn -k uvicorn.workers.UvicornWorker backend.app:app --bind 0.0.0.0:8000
    ```

## Important Notes

*   **Data Generation**: The application requires data to be generated *before* startup. The `Dockerfile` handles this automatically. If deploying manually, you must run the scripts in Step 4 above.
*   **Port**: The `Dockerfile` exposes port `8000`. Cloud providers typically inject a `PORT` environment variable which `uvicorn` or `gunicorn` inside the container will respect (or the mapping handles it).
*   **Production vs Dev**: The `run_flowvision.bat` script is for local development with hot-reload. For deployment, use the Docker or Gunicorn methods.
