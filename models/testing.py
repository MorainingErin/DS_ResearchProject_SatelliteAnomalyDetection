import numpy as np
from utils.dataloader import DataLoader
from utils.visualizer import Visualizer
from utils.evaluator import Evaluator
from utils.classifier import Classifier
from utils.local_threshold import LocalThresholdEvaluator
from models.models import ARIMARunner, XGBoostRunner, LSTMRunner
import torch


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

        # # Plot the results of the best model
        # arima_visr = Visualizer(model_name=f"ARIMA_{element}",
        #                         satellite_name=satellite_name,
        #                         out_path=vis_out_path,
        #                         timestamp=timestamps,
        #                         actual_values=all_data,
        #                         predicted_values=all_predictions, 
        #                         manoeuvres=manoeuvres,
        #                         element=element)
        # arima_visr.save_model_parameter(arima_runner)
        # arima_visr.plot_residual_distributions()
        # arima_visr.plot_predictions()
        # arima_visr.plot_predictions(residual_only=True)
        # arima_visr.plot_violins(day_range=[2])
        # print("ARIMA plotting finished.")

        # Evaluate the model
        arima_evaluator = Evaluator(model_name=f"ARIMA_{element}",
                                    satellite_name=satellite_name,
                                    out_path=vis_out_path,
                                    timestamp=timestamps,
                                    residuals=np.abs(np.array(all_data) - np.array(all_predictions).flatten()),
                                    ground_truth_manoeuvers=manoeuvres,
                                    matching_max_days=3)
        arima_evaluator.compute_all_precision_recall()
        # arima_evaluator.plot_precision_recall_curve()
        print("ARIMA evaluation finished.")

        # # Evaluate the model using local thresholds
        # arima_evaluator = LocalThresholdEvaluator(model_name=f"ARIMA_{element}_local",
        #                             satellite_name=satellite_name,
        #                             out_path=vis_out_path,
        #                             timestamp=timestamps,
        #                             residuals=np.abs(np.array(all_data) - np.array(all_predictions).flatten()),
        #                             ground_truth_manoeuvers=manoeuvres,
        #                             matching_max_days=5)
        # arima_evaluator.compute_all_precision_recall()
        # arima_evaluator.plot_precision_recall_curve()
        # print("ARIMA local evaluation finished.")

        # # Classify the residuals using SVM
        # arima_classifier = Classifier(model_name=f"ARIMA_{element}_svm",
        #                               satellite_name=satellite_name,
        #                               out_path=vis_out_path,
        #                               timestamp=timestamps,
        #                               residuals=np.abs(np.array(all_data) - np.array(all_predictions).flatten()),
        #                               ground_truth_manoeuvers=manoeuvres, 
        #                               matching_max_days=3)

        # nus = [0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.03, 0.01]
        # precision_recalls = []
        # for nu in nus:
        #     arima_classifier.classify_residuals_by_svm(nu)
        #     precision, recall = arima_classifier.compute_simple_matching_precision_recall_for_one_threshold()
        #     precision_recalls.append((nu, precision, recall))
        
        # arima_classifier.plot_precision_recall_curve(precision_recalls)
        # print("ARIMA SVM classification finished.")
        

        # # Load XGBoost model
        # xgboost_runner = XGBoostRunner(model_out_path / element, verbose)
        # xgboost_runner.set_model_name("xgboost-best.pkl")
        # xgboost_runner.load_model()
        # print("Best XGBoost model loaded. Parameter:", xgboost_runner.param_str())

        # # Retrain?
        # # xgboost_runner.train(all_data)
        # diff_all_data = xgboost_runner.difference_data(all_data, difference_order=1)
        # _, diff_all_predictions = xgboost_runner.predict(all_data)

        # # Fix window size for xgboost
        # win_size = diff_all_data.shape[0] - diff_all_predictions.shape[0]

        # # Plot
        # xgboost_visr = Visualizer(model_name=f"XGBoost_{element}",
        #                           satellite_name=satellite_name,
        #                           out_path=vis_out_path,
        #                           timestamp=timestamps[win_size+1:],
        #                           actual_values=diff_all_data[win_size:],
        #                           predicted_values=diff_all_predictions,
        #                           manoeuvres=manoeuvres,
        #                           element=element)
        # xgboost_visr.save_model_parameter(xgboost_runner)
        # xgboost_visr.plot_residual_distributions()
        # xgboost_visr.plot_predictions()
        # xgboost_visr.plot_predictions(residual_only=True)
        # xgboost_visr.plot_violins(day_range=[2])
        # print("XGBoost plotting finished.")

        # # Evaluate the model
        # xgboost_evaluator = Evaluator(model_name=f"XGBoost_{element}",
        #                             satellite_name=satellite_name,
        #                             out_path=vis_out_path,
        #                             timestamp=timestamps[win_size+1:],
        #                             residuals=np.abs(np.array(diff_all_data[win_size:]) - np.array(diff_all_predictions).flatten()),
        #                             ground_truth_manoeuvers=manoeuvres,
        #                             matching_max_days=3)
        # xgboost_evaluator.compute_all_precision_recall()
        # xgboost_evaluator.plot_precision_recall_curve()
        # print("XGBoost evaluation finished.")

 
        # # load LSTM model
        # lstm_runner = LSTMRunner(model_out_path / element, verbose)
        # lstm_runner.set_model_name("lstm-best.pkl")
        # lstm_runner.load_model()
        # print("Best LSTM model loaded. Parameter:", lstm_runner.param_str())

        # # Retrain?
        # # lstm_runner.train(all_data)
        # diff_all_data = lstm_runner.difference_data(all_data, difference_order=1)
        # _, diff_all_predictions = lstm_runner.predict(all_data)

        # # Fix window size for lstm
        # win_size = diff_all_data.shape[0] - diff_all_predictions.shape[0]

        # # Plot
        # lstm_visr = Visualizer(model_name=f"LSTM_{element}",
        #                           satellite_name=satellite_name,
        #                           out_path=vis_out_path,
        #                           timestamp=timestamps[win_size+1:],
        #                           actual_values=diff_all_data[win_size:],
        #                           predicted_values=diff_all_predictions,
        #                           manoeuvres=manoeuvres,
        #                           element=element)
        # lstm_visr.save_model_parameter(lstm_runner)
        # lstm_visr.plot_residual_distributions()
        # lstm_visr.plot_predictions()
        # lstm_visr.plot_predictions(residual_only=True)
        # lstm_visr.plot_violins(day_range=[2])
        # print("LSTM plotting finished.")

        # # Evaluate the model
        # lstm_evaluator = Evaluator(model_name=f"LSTM_{element}",
        #                             satellite_name=satellite_name,
        #                             out_path=vis_out_path,
        #                             timestamp=timestamps[win_size+1:],
        #                             residuals=np.abs(np.array(diff_all_data[win_size:]) - np.array(diff_all_predictions).flatten()),
        #                             ground_truth_manoeuvers=manoeuvres,
        #                             matching_max_days=3)
        # lstm_evaluator.compute_all_precision_recall()
        # lstm_evaluator.plot_precision_recall_curve()
        # print("LSTM evaluation finished.")
        
    pass
