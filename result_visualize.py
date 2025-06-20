import pandas as pd
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio


class ResultVisualizer:
    def __init__(self, result_folder: Path):
        self.result_folder = result_folder

        # Columns of prediction results:
        # [Timestamp, Date, Brouwer mean motion, residual,label]
        self.prediction_res_file = result_folder / f"{result_folder.name}_BMM.csv"
        self.prediction_res = pd.read_csv(self.prediction_res_file)

        # Datetime from Timestamp (%S:%M.%H) and Date (%d/%m/%Y)
        self.prediction_res['Timestamp'] = pd.to_datetime(
            self.prediction_res['Timestamp'] + ' ' + self.prediction_res['Date'], 
            format='%S:%M.%H %d/%m/%Y'
        )
        self.prediction_res['Date'] = pd.to_datetime(
            self.prediction_res['Date'], 
            format='%d/%m/%Y'
        )

        # Columns of manoeuvre results:
        # [manoeuvre_timestamp, label]
        self.manoeuvre_res_file = result_folder / f"manoeuvre_results.csv"
        self.manoeuvre_res = pd.read_csv(self.manoeuvre_res_file)
        self.manoeuvre_res['Timestamp'] = pd.to_datetime(
            self.manoeuvre_res['manoeuvre_timestamp'], 
            format='%Y-%m-%d %H:%M:%S'
        )
        self.manoeuvre_res.drop(columns=['manoeuvre_timestamp'], inplace=True)
        self.manoeuvre_res['Date'] = self.manoeuvre_res['Timestamp'].dt.date

        self.label_colors = {
            'n': 'gray',
            'tp': 'blue',
            'fp': 'red',
            'fn': 'orange'
        }
        self.y_max = self.prediction_res['Brouwer mean motion'].max()
        self.y_min = self.prediction_res['Brouwer mean motion'].min()
        self.y_max += 0.05 * (self.y_max - self.y_min)
        self.y_min -= 0.05 * (self.y_max - self.y_min)

    def _plot_prediction_lines(self, column_name, fig, show_legend=True, secondary_y=False):
        fig.add_trace(
            go.Scatter(
                x=self.prediction_res['Timestamp'],
                y=self.prediction_res[column_name],
                mode='lines',
                name=f"{column_name} prediction",
                line=dict(color='lightgray', width=1),
                showlegend=False
            ),
            secondary_y=secondary_y
        )

        # Add markers
        for label, color in self.label_colors.items():
            df_label = self.prediction_res[self.prediction_res['label'] == label]
            if not df_label.empty:
                fig.add_trace(
                    go.Scatter(
                        x=df_label['Timestamp'],
                        y=df_label[column_name],
                        mode='markers',
                        name=f"{column_name}: {label}",
                        marker=dict(color=color, 
                                    size=4 if label == 'n' else 8),
                        showlegend=show_legend
                    ),
                    secondary_y=secondary_y
                )

    def _plot_manoeuvre_lines(self, fig):
        for label, df_manoeuvre in self.manoeuvre_res.groupby('label'):
            color = self.label_colors.get(label, 'black')
            x_vals, y_vals = [], []
            for ts in df_manoeuvre['Timestamp']:
                x_vals += [ts, ts, None]
                y_vals += [self.y_min, self.y_max, None]
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    mode='lines',
                    name=f"Manoeuvre: {label}",
                    line=dict(color=color, width=2, dash='dash'),
                    showlegend=True
                ),
                secondary_y=False
            )

    def visualize(self):
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Plot brouwer mean motion
        self._plot_prediction_lines('Brouwer mean motion', fig, show_legend=True, secondary_y=False)
        self._plot_prediction_lines('residual', fig, show_legend=True, secondary_y=True)
        self._plot_manoeuvre_lines(fig)

        # Update layout
        fig.update_layout(
            title=f"Results Visualization for {self.result_folder.name}",
            xaxis_title="Timestamp",
            yaxis_title="Brouwer Mean Motion",
            yaxis2=dict(title="Residual", 
                        overlaying="y", 
                        side="right",
                        tickformat=".2e",
                        exponentformat="e"),
            legend_title="Labels",
            hovermode="x unified",
        )

        fig.show()
        pass


if __name__ == '__main__':
    pio.renderers.default = "browser"  # Set default renderer to browser
    result_folder = Path("test/CryoSat-2")
    result_visualizer = ResultVisualizer(result_folder)
    result_visualizer.visualize()
