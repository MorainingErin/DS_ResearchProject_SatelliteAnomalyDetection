import numpy as np
import pandas as pd
from utils.dataloader import DataLoader
from utils.visualizer import Visualizer
from utils.evaluator import Evaluator
from utils.result_processor import ResultProcessor


def evaluate(data_dir, output_dir, satellite_name, verbose):

    orbital_path = data_dir / "orbital_elements" / f"{satellite_name}.csv"
    manoeuvre_path = data_dir / "proc_manoeuvres" / f"{satellite_name}.csv"
    # model_out_path = output_dir / satellite_name / "models"
    # model_out_path.mkdir(parents=True, exist_ok=True)
    res_path = output_dir / satellite_name / "res"
    res_path.mkdir(parents=True, exist_ok=True)
    vis_out_path = output_dir / satellite_name / "vis"
    vis_out_path.mkdir(parents=True, exist_ok=True)

    dataloader = DataLoader(orbital_path, manoeuvre_path)
    dataloader.load_orbital_data()
    dataloader.load_manoeuvre_data()
    elements = dataloader.get_selected_elements()
    manoeuvres = dataloader.get_manoeuvre_data()

    for element in elements:
        # get raw data and predictions
        arima_raw_data = pd.read_csv(res_path / f"{satellite_name}_{element}_arima_raw_data.csv", index_col=0)
        arima_predictions = pd.read_csv(res_path / f"{satellite_name}_{element}_arima_predictions.csv", index_col=0)
        xgboost_diff_data = pd.read_csv(res_path / f"{satellite_name}_{element}_xgboost_diff_data.csv", index_col=0)
        xgboost_predictions = pd.read_csv(res_path / f"{satellite_name}_{element}_xgboost_predictions.csv", index_col=0)
        lstm_diff_data = pd.read_csv(res_path / f"{satellite_name}_{element}_lstm_diff_data.csv", index_col=0)
        lstm_predictions = pd.read_csv(res_path / f"{satellite_name}_{element}_lstm_predictions.csv", index_col=0)

        # # Plot the results of the best ARIMA model
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

        # Evaluate ARIMA model
        process_type = 'local_diff'
        if process_type == 'local_diff':
            local_days = 5
            kernel = [1 / local_days for i in range(local_days)]
        else:
            kernel = None # or 'dynamic' if you want to use dynamic normalization
        res_processor = ResultProcessor(model_name=f"ARIMA_{element}",
                                         satellite_name=satellite_name,
                                         out_path=res_path,
                                         timestamps=arima_predictions.index,
                                         raw_data=arima_raw_data[element],
                                         predictions=arima_predictions['Prediction'],
                                         process_type=process_type,
                                         kernel=kernel)
        timestamps, proc_residuals = res_processor.process_residuals()
        
        arima_evaluator = Evaluator(model_name=f"ARIMA_{element}",
                                    satellite_name=satellite_name,
                                    out_path=vis_out_path,
                                    process_type=process_type,
                                    timestamp=timestamps,
                                    residuals=proc_residuals.values,
                                    ground_truth_manoeuvers=manoeuvres,
                                    matching_max_days=3)
        arima_evaluator.compute_all_precision_recall()
        arima_evaluator.plot_precision_recall_curve()
        print(f"ARIMA evaluation using {process_type} thresholds finished.")


        # # Plot the results of the best XGBoost model
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

        # Evaluate XGBoost model
        process_type = 'local_diff'
        if process_type == 'local_diff':
            local_days = 5
            kernel = [1 / local_days for i in range(local_days)]
        else:
            kernel = None # or 'dynamic' if you want to use dynamic normalization
        res_processor = ResultProcessor(model_name=f"XGBoost_{element}",
                                         satellite_name=satellite_name,
                                         out_path=res_path,
                                         timestamps=xgboost_predictions.index,
                                         raw_data=xgboost_diff_data[element],
                                         predictions=xgboost_predictions['Prediction'],
                                         process_type=process_type,
                                         kernel=kernel)
        timestamps, proc_residuals = res_processor.process_residuals()
        
        xgboost_evaluator = Evaluator(model_name=f"XGBoost_{element}",
                                    satellite_name=satellite_name,
                                    out_path=vis_out_path,
                                    process_type=process_type,
                                    timestamp=timestamps,
                                    residuals=proc_residuals.values,
                                    ground_truth_manoeuvers=manoeuvres,
                                    matching_max_days=3)
        xgboost_evaluator.compute_all_precision_recall()
        xgboost_evaluator.plot_precision_recall_curve()
        print(f"XGBoost evaluation using {process_type} thresholds finished.")


        # # Plot the results of the best LSTM model
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

        # # Evaluate LSTM model
        process_type = 'local_diff'
        if process_type == 'local_diff':
            local_days = 5
            kernel = [1 / local_days for i in range(local_days)]
        else:
            kernel = None # or 'dynamic' if you want to use dynamic normalization
        res_processor = ResultProcessor(model_name=f"LSTM_{element}",
                                         satellite_name=satellite_name,
                                         out_path=res_path,
                                         timestamps=lstm_predictions.index,
                                         raw_data=lstm_diff_data[element],
                                         predictions=lstm_predictions['Prediction'],
                                         process_type=process_type,
                                         kernel=kernel)
        timestamps, proc_residuals = res_processor.process_residuals()
        
        lstm_evaluator = Evaluator(model_name=f"LSTM_{element}",
                                    satellite_name=satellite_name,
                                    out_path=vis_out_path,
                                    process_type=process_type,
                                    timestamp=timestamps,
                                    residuals=proc_residuals.values,
                                    ground_truth_manoeuvers=manoeuvres,
                                    matching_max_days=3)
        lstm_evaluator.compute_all_precision_recall()
        lstm_evaluator.plot_precision_recall_curve()
        print(f"LSTM evaluation using {process_type} thresholds finished.")


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