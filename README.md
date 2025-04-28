# DS_ResearchProject: Satellite Orbital Anomaly Detection

This project focuses on detecting anomalies (often corresponding to satellite manoeuvres) in satellite orbital data using univariate time-series forecasting models.

Forecasting models used:
- **ARIMA**: A linear statistical model suited for stationary time series.
- **XGBoost**: A tree-based ensemble model capable of capturing nonlinear patterns.

The feature used for modeling and anomaly detection is the **Brouwer Mean Motion**, selected through exploratory analysis.

## Project Structure (TODO)

```
├── main.py             # Main entry script to run preprocessing, analysis, training, and testing
├── config.py           # Configuration file defining command-line arguments
├── run.sh              # Bash script for batch processing multiple satellites
├── utils/
│   ├── config.py       # Argument parser (imported by main.py)
│   ├── exploratory_analysis.py  # Functions for exploratory data analysis
│   ├── preprocess_dataset.py    # Functions to preprocess raw satellite data
├── models/
│   ├── training.py     # Functions for training models (ARIMA and XGBoost)
│   ├── testing.py      # Functions for testing models and generating residuals
├── satellite_data/     # Folder to store raw satellite TLE data
├── output/             # Folder to store processed data, model outputs, and plots
├── README.md           # Project description and usage instructions
├── requirements.txt    # Python package dependencies
```

## Key Components

- Based on the selected `--mode`, the `main()` calls:
  - `preprocess_dataset()` to preprocess data
  - `exploratory_analysis()` to visualize orbital data for exploratory analysis
  - `train()` to train ARIMA/XGBoost models, with grid search to find the best hyperparameters
  - `test()` to generate residuals for anomaly detection and plotting

<!-- - **run.sh**: Batch script to automate processing for multiple satellites. It runs preprocessing once, and then analysis, training, and testing for each satellite.

- **config.py**:
  - Defines command-line arguments such as `--mode`, `--data_dir`, `--output_dir`, `--satellite_name`, and `--verbose`. -->

## Setup Instructions

1. Clone the repository:

```bash
git clone https://github.com/MorainingErin/DS_ResearchProject
cd DS_ResearchProject
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Prepare data:

The raw data (`orbital_elements/` and `manoeuvres/`) already included with this project. You can also put your own data under `satellite_data/` folder and modify the file path in code.

4. Run the pipeline manually:

```bash
python main.py --mode preprocess
python main.py --mode analysis --satellite_name "Fengyun-2F"
python main.py --mode train --satellite_name "Fengyun-2F"
python main.py --mode test --satellite_name "Fengyun-2F"
```

Or execute the batch processing script:

```bash
bash ./scripts/run.sh
```


## Future Work

- Extend to multivariate feature modeling
- Incorporate deep learning approaches like LSTM
- Implement automated thresholding for anomaly detection
- Evaluate models quantitatively using precision/recall and ROC curves
