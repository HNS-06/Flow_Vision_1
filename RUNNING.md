# How to Run FlowVision

Follow these steps to set up and run the FlowVision application.

## 1. Install Dependencies

Open your terminal in the project directory (`f:\Projects\flowvision2`) and run:

```bash
pip install -r requirements.txt
```

## 2. Generate Synthetic Data

Generate the initial dataset required for the application:

```bash
python scripts/generate_sample_data.py
```

## 3. Run Machine Learning Pipeline

Run the following scripts in order to train the models and generate necessary artifacts:

```bash
# Preprocess the data
python ml_pipeline/data_preprocessing.py

# Run Leak Detection Model
python ml_pipeline/leak_detection.py

# Run Consumption Forcast Model
python ml_pipeline/consumption_forecast.py
```

## 4. Start the Application

Start the backend server. This will also serve the frontend.

```bash
python -m backend.app
```

The application will be available at: [http://localhost:8000](http://localhost:8000)

## Alternative: One-Click Run (Windows)

You can also try running the batch file, which automates these steps:

```bash
run_flowvision.bat
```
