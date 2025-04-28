# -*- coding = utf-8 -*-

import matplotlib.pyplot as plt
from utils.dataloader import DataLoader


def exploratory_analysis(data_dir, output_dir, satellite_name, verbose=True):
    orbital_path = data_dir / "orbital_elements" / f"{satellite_name}.csv"
    manoeuvre_path = data_dir / "proc_manoeuvres" / f"{satellite_name}.csv"
    orbital_out_path = output_dir / satellite_name / "exp"
    orbital_out_path.mkdir(parents=True, exist_ok=True)

    dataloader = DataLoader(orbital_path, manoeuvre_path)
    selected_elements = dataloader.get_selected_elements()

    processor = DataAnalyser(orbital_out_path=orbital_out_path, dataloader=dataloader, verbose=verbose)
    processor.load_data()
    processor.check_save_info()

    # Draw total distribution
    processor.plot_distribution()

    # Draw seperate features
    processor.plot_orbital_seperate(with_diff=True)

    # Draw batched features
    processor.plot_orbital_batch(data_type="abs")

    # Draw selected features
    processor.plot_orbital_seperate(with_diff=False)
    processor.plot_orbital_batch(elements=selected_elements,
                                 data_type="abs")
    processor.plot_orbital_batch(elements=selected_elements,
                                 data_type="diff")
    

class DataAnalyser:
    def __init__(self, orbital_out_path, dataloader,
                 verbose: bool = True):
        self.orbital_out_path = orbital_out_path
        self.dataloader = dataloader
        self.satellite_name = self.dataloader.satellite_name
        self.orbital_data = None
        self.manoeuvre_data = None
        self.verbose = verbose
    
    def print(self, message: str):
        if self.verbose:
            print(message)

    def load_data(self):
        # self.print(f"Loading data for satellite: {self.satellite_name}...")
        # Load the orbital data
        # self.print(f"Loading orbital data from {self.dataloader.orbital_path}...")
        self.dataloader.load_orbital_data()
        self.orbital_data = self.dataloader.orbital_data
        self.print("Orbital data loaded successfully.")
        
        # Load the manoeuvre data
        # self.print(f"Loading manoeuvre data from {self.dataloader.manoeuvre_path}...")
        self.dataloader.load_manoeuvre_data()
        self.manoeuvre_data = self.dataloader.manoeuvre_data
        self.print("Manoeuvre data loaded successfully.")
        return

    def check_save_info(self):
        # self.print(f"Saving data info for satellite: {self.satellite_name}...")
        with open(self.orbital_out_path / f"{self.satellite_name}_info.txt", "w") as f:
            f.write("Orbital Data Info:\n")
            self.orbital_data.info(buf=f)
            f.write("\nManoeuvre Data Info:\n")
            self.manoeuvre_data.info(buf=f)
        self.print(f"Data info saved to {self.orbital_out_path / f'{self.satellite_name}_info.txt'}.")
        pass

    def plot_distribution(self):
        # self.print(f"Plotting distribution for satellite: {self.satellite_name}...")
        elements = self.orbital_data.columns[1:]  # Exclude the first column (timestamps)
        plt.figure(figsize=(12, 8))  # Increased figure size for better spacing

        for i, element in enumerate(elements):
            ax = plt.subplot(2, 3, i + 1)
            ax.hist(self.orbital_data[element], bins=30, 
                    color="royalblue", alpha=0.7)
            ax.set_title(element)
            plt.xlabel("Value")
            ax.set_ylabel("Frequency")
            ax.grid(True)

            # Rotate x-ticks and use scientific notation if values are small
            ax.tick_params(axis='x', rotation=45)
            ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
            # plt.xticks(rotation=45)  # Rotate x-axis labels for better readability

        # Add a title for the enture figure
        # plt.suptitle(f"Distribution of Orbital Elements for {self.satellite_name}", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 1])

        # Save figure with satellite name
        figure_path = self.orbital_out_path / "total_distribution.png"
        plt.savefig(figure_path, dpi=300, bbox_inches="tight")
        plt.close()

        self.print(f"Distribution plots saved to {figure_path.name}")
        pass

    def _draw_subplot(self, ax, data, data_label, color="blue"):
        ax.plot(self.orbital_data.iloc[:, 0], data, label=data_label, color=color)
        for i, timestamp in enumerate(self.manoeuvre_data["end_timestamp"].dropna()):
            ax.axvline(
                x=timestamp,
                color="darkorange",
                linestyle="--",
                alpha=0.5,
                label="Manoeuvre" if i == 0 else None  # Add label only for the first line
            )
        ax.set_ylabel(f"{data_label}")
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        ax.grid(True)
        ax.legend(loc="upper right")

    def plot_orbital_seperate(self, with_diff=True):
        # self.print(f"Plotting orbital elements seperately for satellite: {self.satellite_name}")
        elements = self.orbital_data.columns[1:]
        fig_num = 2 if with_diff else 1
        for element in elements:
            fig, axes = plt.subplots(fig_num, 1, figsize=(12, 3 * fig_num), sharex=True)
            if not with_diff:
                axes = [axes]
            # Draw original lines
            self._draw_subplot(axes[0], 
                               data=self.orbital_data[element],
                               data_label=element,
                               color="blue")
            if with_diff:
                self._draw_subplot(axes[1],
                                data=self.orbital_data[element].diff().fillna(0),
                                data_label=f"Difference of {element}",
                                color="red")
            axes[-1].set_xlabel("Timestamps")
            fig.tight_layout()
            diff_suffix = "_withdiff" if with_diff else ""
            figure_path = self.orbital_out_path / f"Sep_{element}{diff_suffix}.png"
            fig.savefig(figure_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            self.print(f"Seperate plotting finished. Saved as {figure_path.name}")
        pass

    def plot_orbital_batch(self, elements=None, data_type="abs"):
        """type: abs, diff"""
        elements = self.orbital_data.columns[1:] if elements is None else elements
        element_str = "-".join([x[:3] for x in map(str, elements)])
        
        fig, axes = plt.subplots(len(elements), 1,
                                 figsize=(12, 3 * len(elements)),
                                 sharex=True)
        for i, element in enumerate(elements):
            if data_type == "abs":
                self._draw_subplot(axes[i],
                                   data=self.orbital_data[element],
                                   data_label=element,
                                   color="blue")
            elif data_type == "diff":
                self._draw_subplot(axes[i],
                                   data=self.orbital_data[element].diff().fillna(0),
                                   data_label=f"Difference of {element}",
                                   color="red")
            else:
                raise NotImplementedError(f"Error data_type: {data_type}")
        axes[-1].set_xlabel("Timestamps")
        fig.tight_layout()
        figure_path = self.orbital_out_path / f"Batched_{data_type}_{element_str}.png"
        fig.savefig(figure_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        self.print(f"Batched plotting finished. Saved as {figure_path.name}")

    def plot_orbital_all(self):
        # self.print(f"Plotting orbital elements for satellite: {self.satellite_name}...")
        elements = self.orbital_data.columns[1:]

        # Total version
        fig, axes = plt.subplots(len(elements), 1, 
                                 figsize=(12, 3 * len(elements)),
                                 sharex=True)
        for i, element in enumerate(elements):
            self._draw_subplot(axes[i], 
                               data=self.orbital_data[element],
                               data_label=element, 
                               color="blue")
        axes[-1].set_xlabel("Timestamps")
        # fig.suptitle(f"Orbital elements for {self.satellite_name} with manoeuvres", fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 1])
        figure_path = self.orbital_out_path / f"{self.satellite_name}_all_element.png"
        fig.savefig(figure_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        self.print(f"All orbital element plots for {self.satellite_name} saved.")
        pass
