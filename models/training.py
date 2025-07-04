from utils.dataloader import DataLoader
from models.models import ARIMARunner, XGBoostRunner, LSTMRunner
import torch
from tqdm import tqdm
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning


def grid_search(model_class, param_grid, train_data, test_data):

    best_score = float("inf")
    best_runner = None
    pbar = tqdm(total=len(param_grid))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        warnings.simplefilter("ignore", category=UserWarning)
        for params in param_grid:
            # Initialize the model with the current parameter combination
            model = model_class(**params)
            
            model.set_model_name()
            pbar.set_description(f"{model.param_str()}, min_score={best_score:.2e}")

            if train_data is not None:
                model.train(train_data)

            score, _ = model.predict(test_data)  # Return: score, predictions

            if score < best_score:
                best_score = score
                best_runner = model
            
            pbar.update(1)

    return best_score, best_runner


def train(data_dir, output_dir, satellite_name, verbose):

    orbital_path = data_dir / "orbital_elements" / f"{satellite_name}.csv"
    manoeuvre_path = data_dir / "proc_manoeuvres" / f"{satellite_name}.csv"
    model_out_path = output_dir / satellite_name / "models"
    model_out_path.mkdir(parents=True, exist_ok=True)

    dataloader = DataLoader(orbital_path, manoeuvre_path)
    dataloader.load_orbital_data()
    dataloader.load_manoeuvre_data()
    elements = dataloader.get_selected_elements()


    for element in elements:
        # get train, test data and all_data
        train_data, test_data = dataloader.get_train_test_data(element)
        all_data = dataloader.get_all_data(element)
        (model_out_path / element).mkdir(parents=True, exist_ok=True)

        # ARIMA Model Grid Search
        arima_param_grid = [
            {
                "model_folder": model_out_path / element,
                "verbose": verbose,
                "order": (p, d, q)
            } 
            for p in range(0, 6) 
            for d in range(0, 3) 
            for q in range(0, 6)
        ]
        best_arima_score, best_arima_runner = grid_search(
            model_class=ARIMARunner,
            param_grid=arima_param_grid,
            train_data=None,
            test_data=all_data
        )
        best_arima_runner.set_model_name(f"arima-best.pkl")
        best_arima_runner.save_model()
        print(f"Best ARIMA params for {element}: {best_arima_runner.param_str()} with an AIC of {best_arima_score}")


        # XGBoost Model Grid Search
        xgb_param_grid = [
            {
                "model_folder": model_out_path / element,
                "verbose": verbose,
                "window": w,
                "n_estimators": n, 
                "max_depth": d,
                "learning_rate": lr,
            }
            for w in [3, 5, 10, 20]
            for n in [10, 50, 100, 200]
            for d in [3, 5, 7]
            for lr in [0.01, 0.05, 0.1, 0.2]
        ]
        best_xgb_score, best_xgb_runner = grid_search(
            model_class=XGBoostRunner,
            param_grid=xgb_param_grid,
            train_data=train_data,
            test_data=test_data
        )
        best_xgb_runner.set_model_name(f"xgboost-best.pkl")
        best_xgb_runner.save_model()
        print(f"Best XGBoost params for {element}: {best_xgb_runner.param_str()} with a score of {best_xgb_score}")


        # LSTM Model Grid Search
        lstm_param_grid = [
            {
                "model_folder": model_out_path / element,
                "verbose": verbose,
                "window": w,
                "hidden_dim": h,
                "num_layers": l,
                "learning_rate": lr
            }
            for w in [5, 10, 20]
            for h in [16, 32, 64]
            for l in [1, 2]
            for lr in [0.001, 0.003, 0.01, 0.03]
        ]

        best_lstm_score, best_lstm_runner = grid_search(
            model_class=LSTMRunner,
            param_grid=lstm_param_grid,
            train_data=train_data,
            test_data=test_data
        )
        best_lstm_runner.set_model_name(f"lstm-best.pkl")
        best_lstm_runner.save_model()
        print(f"Best LSTM params for {element}: {best_lstm_runner.param_str()} with a score of {best_lstm_score}")

    pass
