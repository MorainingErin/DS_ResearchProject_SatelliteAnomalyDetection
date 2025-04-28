from utils.dataloader import DataLoader
from utils.visualizer import Visualizer
from utils.evaluator import Evaluator
from models.models import ARIMARunner, XGBoostRunner


def test(data_dir, output_dir, satellite_name, verbose):

    orbital_path = data_dir / "orbital_elements" / f"{satellite_name}.csv"
    manoeuvre_path = data_dir / "proc_manoeuvres" / f"{satellite_name}.csv"
    model_out_path = output_dir / satellite_name / "models"
    model_out_path.mkdir(parents=True, exist_ok=True)
    vis_out_path = output_dir / satellite_name / "vis"
    vis_out_path.mkdir(parents=True, exist_ok=True)

    dataloader = DataLoader(orbital_path, manoeuvre_path)
    dataloader.load_orbital_data()
    dataloader.load_manoeuvre_data()
    elements = dataloader.get_selected_elements()
    manoeuvres = dataloader.get_manoeuvre_data()

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

        # Plot the results of the best model
        arima_visr = Visualizer(model_name=f"ARIMA_{element}",
                                satellite_name=satellite_name,
                                out_path=vis_out_path,
                                timestamp=timestamps,
                                actual_values=all_data,
                                predicted_values=all_predictions, 
                                manoeuvres=manoeuvres,
                                element=element)
        arima_visr.save_model_parameter(arima_runner)
        arima_visr.plot_residual_distributions()
        arima_visr.plot_predictions()
        arima_visr.plot_predictions(residual_only=True)
        arima_visr.plot_violins(day_range=[2])
        print("ARIMA plotting finished.")
        # visualizer.plot_residuals_with_manoeuvres()

        # Load XGBoost model
        xgboost_runner = XGBoostRunner(model_out_path / element, verbose)
        xgboost_runner.set_model_name("xgboost-best.pkl")
        xgboost_runner.load_model()
        print("Best XGBoost model loaded. Parameter:", xgboost_runner.param_str())

        # Retrain?
        # xgboost_runner.train(all_data)
        _, all_predictions = xgboost_runner.predict(all_data)

        # Fix window size for xgboost
        win_size = all_data.shape[0] - all_predictions.shape[0]

        # Plot
        xgboost_visr = Visualizer(model_name=f"XGBoost_{element}",
                                  satellite_name=satellite_name,
                                  out_path=vis_out_path,
                                  timestamp=timestamps[win_size:],
                                  actual_values=all_data[win_size:],
                                  predicted_values=all_predictions,
                                  manoeuvres=manoeuvres,
                                  element=element)
        xgboost_visr.save_model_parameter(xgboost_runner)
        xgboost_visr.plot_residual_distributions()
        xgboost_visr.plot_predictions()
        xgboost_visr.plot_predictions(residual_only=True)
        xgboost_visr.plot_violins(day_range=[2])
        print("XGBoost plotting finished.")

    pass
