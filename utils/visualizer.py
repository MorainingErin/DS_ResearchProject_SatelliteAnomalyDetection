import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class Visualizer:
    def __init__(self, model_name, satellite_name, out_path,
                 timestamp, actual_values, predicted_values, manoeuvres, 
                 param_str=None, element=None):

        self.model_name = model_name
        self.satellite_name = satellite_name
        self.out_path = out_path

        self.timestamp = timestamp
        self.actual_values = np.array(actual_values)
        self.predicted_values = np.array(predicted_values).flatten()
        self.manoeuvres = manoeuvres
        
        self.param_str = param_str
        self.element = element
        # self.window = self.actual_values.shape[0] - self.predicted_values.shape[0]


    def plot_residual_distributions(self):
        residuals = np.abs(self.actual_values - self.predicted_values)

        fig = plt.figure(figsize=(4, 2.5))
        ax = plt.subplot()
        sns.kdeplot(residuals, ax=ax, color='orange', fill=True, alpha=0.6)
        # ax.set_title('Residuals')
        ax.set_xlabel('Residuals', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        ax.grid(True)

        fig.tight_layout()
        fig.savefig(self.out_path / f'{self.model_name}_residuals_distribution.png')
        plt.close(fig)


    def plot_predictions(self, residual_only=False):
        """
        Plot the actual and predicted values for both training and test sets.
        """
        residuals = np.abs(self.actual_values - self.predicted_values)

        fig_num = 1 if residual_only else 2
        fig, axes = plt.subplots(fig_num, 1, figsize=(12, 3 * fig_num), sharex=True)

        ax1 = None if residual_only else axes[0]
        ax2 = axes if residual_only else axes[1]

        # Top: Actual v.s. Predicted
        if ax1 is not None:
            ax1.plot(self.timestamp, self.actual_values, label="Actual", color="black")
            ax1.plot(self.timestamp, self.predicted_values, label="Predicted", color="dodgerblue")
            for i, timestamp in enumerate(self.manoeuvres["end_timestamp"].dropna()):
                ax1.axvline(
                    x=timestamp,
                    color="darkorange",
                    linestyle="--",
                    alpha=0.5,
                    label="Manoeuvre" if i == 0 else None  # Add label only for the first line
                )
            ax1.set_ylabel(self.element)
            ax1.legend()
            ax1.grid(True)

        # Bottom: Residuals
        ax2.plot(self.timestamp, residuals, label="Residuals", color="red")
        for i, timestamp in enumerate(self.manoeuvres["end_timestamp"].dropna()):
            ax2.axvline(
                x=timestamp,
                color="darkorange",
                linestyle="--",
                alpha=0.5,
                label="Manoeuvre" if i == 0 else None  # Add label only for the first line
            )
        ax2.set_ylabel(f"Absolute difference")
        ax2.set_xlabel(f"Timestamps")
        ax2.legend()
        ax2.grid(True)

        fig.tight_layout()
        fig_suffix = "_only" if residual_only else "_curve"
        fig.savefig(self.out_path / f'{self.model_name}_residual{fig_suffix}.png')
        plt.close(fig)

        # plt.figure(figsize=(12, 8))
        # # # Plot training data
        # plt.plot(self.timestamp[self.window:], self.actual_values[self.window:], label='Actual Values', color='blue', linewidth=2)
        # plt.plot(self.timestamp[self.window:], self.predicted_values, label='Predicted Values', color='orange', linestyle='--', linewidth=2)

        # # plt.title(f'{self.model_name} Predictions for {self.element} of {self.satellite_name}', fontsize=14)
        # plt.title(f'{self.model_name} Predictions for {self.element} ({self.satellite_name})', fontsize=14)
        # plt.xlabel('Timestamps', fontsize=12)
        # plt.ylabel('Values', fontsize=12)
        # plt.legend()
        # plt.grid(True)
        # plt.tight_layout()
        # plt.savefig(self.orbital_out_path / f'{self.model_name}_predictions__{self.element} ({self.satellite_name}).png')


    def plot_violins(self, day_range=range(1, 2)):
        total_residuals = np.abs(self.actual_values - self.predicted_values)
        df = pd.DataFrame({
            "residual": list(total_residuals),
            "group": ["Total"] * len(total_residuals)
        })

        for day in day_range:
            within_window = None
            for ts in self.manoeuvres["end_timestamp"].dropna():
                window = (
                    (self.timestamp > ts - pd.Timedelta(days=day)) 
                    & (self.timestamp < ts + pd.Timedelta(days=day))
                )
                if within_window is None:
                    within_window = window
                else:
                    within_window = within_window | window
            neighbor_residuals = total_residuals[within_window]
            df = pd.concat([
                df,
                pd.DataFrame({
                    "residual": list(neighbor_residuals),
                    "group": f"Neighbor_{day}" if len(day_range) > 1 else "Neighbor"
                })
            ], ignore_index=True)

        fig, ax = plt.subplots(figsize=(4, 2.5))
        sns.violinplot(data=df, 
                       x="group", 
                       y="residual",
                       hue="group",
                       ax=ax,
                       inner="box",
                       legend=False,
                       palette="Set2")
        ax.set_ylabel("Residual")
        ax.set_xlabel("")
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(self.out_path / f"{self.model_name}_violins.png")
        plt.close(fig)


    # def plot_residuals_with_manoeuvres(self):
    #     """
    #     Plot the residuals (absolute difference between predicted and actual values)
    #     for both training and test sets, and draw manoeuvres as vertical lines.
    #     """
    #     residuals = np.abs(self.predicted_values - self.actual_values[self.window: ])

    #     plt.figure(figsize=(12, 8))
    #     # Plot training residuals
    #     plt.plot(self.timestamp[self.window:],
    #              residuals, label='Residuals (Train)', color='blue', linewidth=2)

    #     # Draw manoeuvres as vertical lines
    #     for timestamp in self.manoeuvres["end_timestamp"]:
    #         plt.axvline(
    #         x=timestamp,
    #         color="darkorange",
    #         linestyle="--",
    #         alpha=1.0,
    #         label="Manoeuvre"
    #         if "Manoeuvre" not in plt.gca().get_legend_handles_labels()[1]
    #         else None,
    #     )

    #     # plt.title(f'{self.model_name} Residuals of {self.element} for {self.satellite_name}', fontsize=14)
    #     plt.title(f'Residuals with Manoeuvres: {self.model_name} - {self.element} ({self.satellite_name})', fontsize=14)
    #     plt.xlabel('Timestamps', fontsize=12)
    #     plt.ylabel('Residuals', fontsize=12)
    #     plt.legend()
    #     plt.grid(True)
    #     plt.tight_layout()
    #     plt.savefig(self.orbital_out_path / f'{self.model_name}_residuals_{self.element} ({self.satellite_name}).png')


    def save_model_parameter(self, model_runner):
        with open(self.out_path / f"{self.model_name}_info.txt", "w") as f:
            f.write(str(model_runner))
            f.write(f"\nPredicted MAE: {self.compute_mae()}")


    def compute_mae(self):
        residuals = np.abs(self.actual_values - self.predicted_values)
        return np.mean(residuals)


    # def display_model_info(self):
    #     """
    #     Display the model name, hyperparameters, and satellite name.
    #     """
    #     print(f"Model Name: {self.model_name}")
    #     print(f"Satellite Name: {self.satellite_name}")
    #     print("Hyperparameters:")
    #     for key, value in self.hyperparameters.items():
    #         print(f"  {key}: {value}")
