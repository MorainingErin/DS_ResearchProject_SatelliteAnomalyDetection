# DS_ResearchProject: Satellite Orbital Anomaly Detection

This project focuses on detecting anomalies in satellite orbital data, often corresponding to satellite manoeuvres, using univariate time-series forecasting models.

## Forecasting Models
- **ARIMA**: A linear statistical model suited for stationary time series.
- **XGBoost**: A tree-based ensemble model capable of capturing nonlinear patterns.

The primary feature used for modeling and anomaly detection is the **Brouwer Mean Motion**, selected through exploratory analysis.

---

## Project Structure

```
├── main.py             # Main entry script to run preprocessing, analysis, training, and testing
├── scripts/            # Scripts for batch processing and file management
│   ├── run.sh          # Bash script for batch processing multiple satellites
│   ├── run.ps1         # PowerShell script for batch processing on Windows
│   ├── move_files.py   # Script to organize and move output files
├── utils/              # Utility modules for data handling and analysis
│   ├── config.py       # Argument parser for command-line options
│   ├── dataloader.py   # DataLoader class for loading and splitting datasets
│   ├── data_processor.py # Preprocessing raw satellite data
│   ├── data_analyser.py  # Exploratory analysis and visualization
|   ├── evaluator.py    # Module for quantitative evaluation for later use
|   ├── visualizer.py   # Module for visualizing experiment results
│   ├── __init__.py     # Module initializer
├── models/             # Model training and testing modules
│   ├── models.py       # Model classes for ARIMA, XGBoost, and LSTM
│   ├── training.py     # Training pipeline with grid search for hyperparameter tuning
│   ├── testing.py      # Testing pipeline for predictions and visualization
├── satellite_data/     # Folder to store raw satellite TLE and manoeuvre data
├── output/             # Folder to store processed data, model outputs, and plots
├── README.md           # Project description and usage instructions
├── requirements.txt    # Python package dependencies
```

---

## Key Components

Based on the selected `--mode`, the `main()` function calls:
- `preprocess_dataset()` to preprocess raw satellite data.
- `exploratory_analysis()` to visualize orbital data for exploratory analysis.
- `train()` to train ARIMA/XGBoost models with grid search for hyperparameter optimization.
- `test()` to generate residuals for anomaly detection and create visualizations.

---

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/MorainingErinDS_ResearchProject_SatelliteAnomalyDetection
cd DS_ResearchProject_SatelliteAnomalyDetection
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Prepare Data
Ensure the raw data (`orbital_elements/` and `manoeuvres/`) is placed in the `satellite_data/` folder. You can also use the provided data or modify the file paths in the code to suit your dataset.


## Running the Pipeline

### Manual Execution
Run the pipeline step-by-step:
```bash
python main.py --mode preprocess
python main.py --mode analysis --satellite_name "Fengyun-2F"
python main.py --mode train --satellite_name "Fengyun-2F"
python main.py --mode test --satellite_name "Fengyun-2F"
```

### Batch Processing
Use the provided batch processing script:
- **Linux/Mac**:
  ```bash
  bash ./scripts/run.sh
  ```
- **Windows**:
  ```powershell
  ./scripts/run.ps1
  ```

---

## Future Work

- Experiment with manual differencing or transformations to improve model performance.
- Extend the pipeline to multivariate feature modeling.
- Incorporate deep learning approaches like LSTM for anomaly detection.
- Implement automated thresholding for anomaly detection.
- Evaluate models quantitatively using precision/recall and ROC curves.
