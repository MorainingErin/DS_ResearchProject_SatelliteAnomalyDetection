import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler


class Classifier:
    def __init__(self, model_name, satellite_name, out_path,
                 timestamp, residuals, ground_truth_manoeuvers, matching_max_days=1):

        self.model_name = model_name
        self.satellite_name = satellite_name
        self.out_path = out_path

        self.residuals = pd.Series(data=residuals, index=pd.to_datetime(timestamp))
        self.ground_truth_manoeuvers = pd.to_datetime(ground_truth_manoeuvers['end_timestamp'])
        self.matching_max_days = matching_max_days
        self.dict_predictions_to_ground_truth = {}
        self.dict_ground_truth_to_predictions = {}

        self.svm_classifier = None
        self.svm_scaler = None
        self.svm_pred_labels = None
        self.svm_pred_raw = None
        self.precision_recall_thresholds = []

    def convert_timestamp_series_to_epoch(self, series):
        return (
            (series - pd.Timestamp(year=1970, month=1, day=1)) // pd.Timedelta(seconds=1)
        ).values

    def classify_residuals_by_svm(self, n):

        # Prepare features: residuals as 1D feature
        X = self.residuals.values.reshape(-1, 1)

        # Standardize features for SVM
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train One-Class SVM (unsupervised)
        clf = OneClassSVM(kernel='rbf', nu=n, gamma='scale')  # nu can be tuned
        clf.fit(X_scaled)

        # Predict: -1 for anomaly, 1 for normal
        y_pred = clf.predict(X_scaled)
        # Convert to 1 for anomaly, 0 for normal for consistency
        y_pred_bin = (y_pred == -1).astype(int)

        # Store results for further analysis
        self.svm_classifier = clf
        self.svm_scaler = scaler
        self.svm_pred_labels = y_pred_bin
        self.svm_pred_raw = y_pred

    def compute_simple_matching_precision_recall_for_one_threshold(self):
        self.dict_predictions_to_ground_truth = {}
        self.dict_ground_truth_to_predictions = {}
        
        matching_max_distance_seconds = pd.Timedelta(days=self.matching_max_days).total_seconds()

        manoeuvre_timestamps_seconds = self.convert_timestamp_series_to_epoch(self.ground_truth_manoeuvers)
        pred_time_stamps_seconds = self.convert_timestamp_series_to_epoch(self.residuals.index)
        predictions = self.residuals.to_numpy()

        for i in range(predictions.shape[0]):
            if self.svm_pred_labels[i] == 1:
                left_index = np.searchsorted(
                    manoeuvre_timestamps_seconds, pred_time_stamps_seconds[i]
                )

                if left_index != 0:
                    left_index -= 1

                index_of_closest = left_index

                if (left_index < self.ground_truth_manoeuvers.shape[0] - 1) and (
                    abs(manoeuvre_timestamps_seconds[left_index] - pred_time_stamps_seconds[i])
                    > abs(manoeuvre_timestamps_seconds[left_index + 1] - pred_time_stamps_seconds[i])
                ):
                    index_of_closest = left_index + 1

                diff = abs(manoeuvre_timestamps_seconds[index_of_closest] - pred_time_stamps_seconds[i])

                if diff < matching_max_distance_seconds:
                    self.dict_predictions_to_ground_truth[i] = (
                        index_of_closest,
                        diff,
                    )
                    if index_of_closest in self.dict_ground_truth_to_predictions:
                        self.dict_ground_truth_to_predictions[index_of_closest].append(i)
                    else:
                        self.dict_ground_truth_to_predictions[index_of_closest] = [i]

        positive_prediction_indices = np.argwhere(self.svm_pred_labels == 1)[:, 0]
        list_false_positives = [
            pred_ind for pred_ind in positive_prediction_indices if pred_ind not in self.dict_predictions_to_ground_truth.keys()
        ]
        list_false_negatives = [
            true_ind for true_ind in np.arange(0, len(self.ground_truth_manoeuvers))
            if true_ind not in self.dict_ground_truth_to_predictions.keys()
        ]

        precision = len(self.dict_ground_truth_to_predictions) / (len(self.dict_ground_truth_to_predictions) + len(list_false_positives))
        recall = len(self.dict_ground_truth_to_predictions) / (len(self.dict_ground_truth_to_predictions) + len(list_false_negatives))

        print("The precision is: ", precision)
        print("The recall is: ", recall)
        return precision, recall

    # def compute_all_precision_recall(self):
    #     nus = [0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.03, 0.01]

    #     for nu in nus:
    #         precision, recall = self.compute_simple_matching_precision_recall_for_one_threshold()
    #         self.precision_recall_thresholds.append((nu, precision, recall))

    #     return self.precision_recall_thresholds

    def plot_precision_recall_curve(self, precision_recalls):
        """
        Plot the precision-recall curve for different thresholds.
        """
        precisions = [item[1] for item in precision_recalls]
        recalls = [item[2] for item in precision_recalls]

        plt.figure(figsize=(8, 6))
        plt.plot(recalls, precisions, marker='o', label="Precision-Recall Curve")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend()
        plt.grid()
        plt.xlim(0.0, 1.0)
        plt.ylim(0.0, 1.0)

        plt.savefig(self.out_path / f'{self.model_name}_Precision_Recall_Curve.png')
        plt.close()