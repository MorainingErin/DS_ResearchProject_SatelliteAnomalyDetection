import pandas as pd
from utils.dataloader import DataLoader
from models.models import ARIMARunner, XGBoostRunner, LSTMRunner


def test(data_dir, output_dir, satellite_name, verbose):

    orbital_path = data_dir / "orbital_elements" / f"{satellite_name}.csv"
    manoeuvre_path = data_dir / "proc_manoeuvres" / f"{satellite_name}.csv"
    model_out_path = output_dir / satellite_name / "models"
    model_out_path.mkdir(parents=True, exist_ok=True)
    res_out_path = output_dir / satellite_name / "res"
    res_out_path.mkdir(parents=True, exist_ok=True)
    vis_out_path = output_dir / satellite_name / "vis"
    vis_out_path.mkdir(parents=True, exist_ok=True)

    dataloader = DataLoader(orbital_path, manoeuvre_path)
    dataloader.load_orbital_data()
    dataloader.load_manoeuvre_data()
    elements = dataloader.get_selected_elements()

    for element in elements:
        # get train, test data and all_data
        all_data = dataloader.get_all_data(element)
        timestamps = dataloader.get_all_timestamps()

        # Load ARIMA model
        arima_runner = ARIMARunner(model_out_path / element, verbose)
        arima_runner.set_model_name(f"arima-best.pkl")
        arima_runner.load_model()
        print("Best ARIMA model loaded. Parameter:", arima_runner.param_str())
        
        # Get the prediction across the whole dataset
        _, all_predictions = arima_runner.predict(all_data)
        # Save the predictions to a CSV file with timestamps
        all_predictions_df = pd.DataFrame({"Timestamp": timestamps, "Prediction": all_predictions})
        all_data_df = pd.DataFrame({"Timestamp": timestamps, element: all_data})
        all_data_df.to_csv(res_out_path / f"{satellite_name}_{element}_arima_raw_data.csv", index=False)
        print("ARIMA raw data saved to CSV.")
        all_predictions_df.to_csv(res_out_path / f"{satellite_name}_{element}_arima_predictions.csv", index=False)
        print("ARIMA predictions saved to CSV.")


        # Load XGBoost model
        xgboost_runner = XGBoostRunner(model_out_path / element, verbose)
        xgboost_runner.set_model_name("xgboost-best.pkl")
        xgboost_runner.load_model()
        print("Best XGBoost model loaded. Parameter:", xgboost_runner.param_str())

        # xgboost_runner.train(all_data)        # Uncomment to retrain the model
        diff_all_data = xgboost_runner.difference_data(all_data, difference_order=1)
        _, diff_all_predictions = xgboost_runner.predict(all_data)
        # Adjust the shape of the timestamps to match the predictions
        win_size = diff_all_data.shape[0] - diff_all_predictions.shape[0]
        # Save the differenced data and predictions to a CSV file with timestamps
        diff_all_predictions_df = pd.DataFrame({"Timestamp": timestamps[win_size+1:], 
                                                "Prediction": diff_all_predictions.flatten()})
        diff_all_data_df = pd.DataFrame({"Timestamp": timestamps[win_size+1:], element: diff_all_data[win_size:]})
        diff_all_data_df.to_csv(res_out_path / f"{satellite_name}_{element}_xgboost_diff_data.csv", index=False)
        print("XGBoost difference data saved to CSV.")
        diff_all_predictions_df.to_csv(res_out_path / f"{satellite_name}_{element}_xgboost_predictions.csv", index=False)
        print("XGBoost predictions saved to CSV.")


        # load LSTM model
        lstm_runner = LSTMRunner(model_out_path / element, verbose)
        lstm_runner.set_model_name("lstm-best.pkl")
        lstm_runner.load_model()
        print("Best LSTM model loaded. Parameter:", lstm_runner.param_str())

        # lstm_runner.train(all_data)       # Uncomment to retrain the model
        diff_all_data = lstm_runner.difference_data(all_data, difference_order=1)
        _, diff_all_predictions = lstm_runner.predict(all_data)
        # Adjust the shape of the timestamps to match the predictions
        win_size = diff_all_data.shape[0] - diff_all_predictions.shape[0]
        # Save the differenced data and predictions to a CSV file with timestamps
        diff_all_predictions_df = pd.DataFrame({"Timestamp": timestamps[win_size+1:], 
                                                "Prediction": diff_all_predictions.flatten()})
        diff_all_data_df = pd.DataFrame({"Timestamp": timestamps[win_size+1:], element: diff_all_data[win_size:]})
        diff_all_data_df.to_csv(res_out_path / f"{satellite_name}_{element}_lstm_diff_data.csv", index=False)
        print("LSTM difference data saved to CSV.")
        diff_all_predictions_df.to_csv(res_out_path / f"{satellite_name}_{element}_lstm_predictions.csv", index=False)
        print("LSTM predictions saved to CSV.")