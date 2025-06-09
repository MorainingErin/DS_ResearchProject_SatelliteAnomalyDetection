# -*- coding = utf-8 -*-

import pandas as pd
from datetime import datetime as dt, timedelta as td
from tqdm import tqdm


def preprocess_dataset(data_dir, verbose=True):

    raw_mano_dir = data_dir / "manoeuvres"
    proc_mano_dir = data_dir / "proc_manoeuvres"
    proc_mano_dir.mkdir(parents=True, exist_ok=True)

    name_dict = {
        "cs2man.txt":"CryoSat-2.csv",
        "h2aman.txt":"Haiyang-2A.csv",
        "ja1man.txt":"Jason-1.csv",
        "ja2man.txt":"Jason-2.csv",
        "ja3man.txt":"Jason-3.csv",
        "s3aman.txt":"Sentinel-3A.csv",
        "s3bman.txt":"Sentinel-3B.csv",
        "s6aman.txt":"Sentinel-6A.csv",
        "srlman.txt":"SARAL.csv",
        "topman.txt":"TOPEX.csv",

        "manFY2D.txt.fy":"Fengyun-2D.csv",
        "manFY2E.txt.fy":"Fengyun-2E.csv",
        "manFY2F.txt.fy":"Fengyun-2F.csv",
        "manFY2H.txt.fy":"Fengyun-2H.csv",
        "manFY4A.txt.fy":"Fengyun-4A.csv",
    }

    for raw_name, proc_name in tqdm(name_dict.items(), desc="Pre-processing"):
        raw_file_path = raw_mano_dir / raw_name
        proc_file_path = proc_mano_dir / proc_name

        fy_flag, expected_columns = False, None
        if "manFY" in raw_file_path.stem:
            fy_flag = True
            expected_columns = 6
        elif "topman" in raw_file_path.stem:
            expected_columns = 9
        else:
            expected_columns = 26
        
        pre_processor = DataPreProcessor(
            raw_file_path=raw_file_path,
            dst_file_path=proc_file_path,
            fy_flag=fy_flag,
            expected_columns=expected_columns
        )
        pre_processor.run()
    
    pass


class DataPreProcessor:

    def __init__(self, raw_file_path, dst_file_path, fy_flag, expected_columns):
        self.raw_file_path = raw_file_path
        self.dst_file_path = dst_file_path
        self.fy_flag = fy_flag
        self.expected_columns = expected_columns
        self.raw_manoeuvre = None
        self.proc_manoeuvre = None
    
    def run(self):
        # Load
        self.load_file()

        # Clean
        if self.fy_flag:
            self.clean_fy_manoeuvre()
        else:
            self.clean_manoeuvre()

        # Save
        self.save_cleaned_file()
        pass

    def load_file(self):
        # Load .txt file (columns are space-separated)
        self.raw_manoeuvre = pd.read_csv(self.raw_file_path,
                                         sep='\s+',
                                         header=None,
                                         usecols=range(self.expected_columns),
                                         engine='python')
        return self.raw_manoeuvre

    def clean_manoeuvre(self):
        # Select relevant columns (adjust these indices based on actual file structure)
        manoeuvre = self.raw_manoeuvre
        if "topman" in self.raw_file_path.name:
            manoeuvre = manoeuvre.iloc[:, [1, 2, 3, 4, 5, 6, 7, 8]].copy()

            # Rename columns for clarity
            manoeuvre.columns = ["Start_Year", "Start_DOY", "Start_Hour", "Start_Min",
                            "End_Year", "End_DOY", "End_Hour", "End_Min"]
        else:
            if "sp" in self.raw_file_path.name:
                manoeuvre = manoeuvre.iloc[:, [1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14, 15, 16]].copy()
            else:
                manoeuvre = manoeuvre.iloc[:, [1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15]].copy()

            # Rename columns for clarity
            manoeuvre.columns = ["Start_Year", "Start_DOY", "Start_Hour", "Start_Min",
                                "End_Year", "End_DOY", "End_Hour", "End_Min",
                                "Med_Year", "Med_DOY", "Med_Hour", "Med_Min", "Med_Sec"]

        # Apply time format conversion
        manoeuvre["start_timestamp"] = manoeuvre.apply(
            lambda x: convert_to_datetime(
                x["Start_Year"], x["Start_DOY"], x["Start_Hour"], x["Start_Min"], "0.000"
            ), axis=1
        )
        manoeuvre["end_timestamp"] = manoeuvre.apply(
            lambda x: convert_to_datetime(
                x["End_Year"], x["End_DOY"], x["End_Hour"], x["End_Min"], "0.000"
            ), axis=1
        )

        if "topman" in self.raw_file_path.name:
            manoeuvre["median_timestamp"] = manoeuvre.apply(
                lambda x: calculate_median(
                    x["start_timestamp"], x["end_timestamp"]
                ), axis=1
            )
        else:
            manoeuvre["median_timestamp"] = manoeuvre.apply(
                lambda x: convert_to_datetime(
                    x["Med_Year"], x["Med_DOY"], x["Med_Hour"], x["Med_Min"], x["Med_Sec"]
                ), axis=1
            )

        # Keep only the timestamps for ground truth
        self.proc_manoeuvre = manoeuvre[["start_timestamp", "end_timestamp", "median_timestamp"]].copy()
        return self.proc_manoeuvre

    def clean_fy_manoeuvre(self):
        # Select relevant columns (adjust these indices based on actual file structure)
        manoeuvre = self.raw_manoeuvre
        manoeuvre = manoeuvre.iloc[:, [0, 1, 2, 4]].copy()

        # Rename columns for clarity
        manoeuvre.columns = ["operation_name", "launch_time", "start_time_cst", "end_time_cst"]

        # Convert CST time to UTC time
        def convert_cst_to_utc(cst_time_str):
            cst_time_str = cst_time_str.replace("/", "-")
            cst_time = dt.strptime(cst_time_str[1:], "%Y-%m-%dT%H:%M:%S")
            utc_time = cst_time - td(hours=8)
            return utc_time

        # Apply conversion
        manoeuvre['start_timestamp'] = manoeuvre.apply(
            lambda x: convert_cst_to_utc(x["start_time_cst"]), axis=1
        )
        manoeuvre['end_timestamp'] = manoeuvre.apply(
            lambda x: convert_cst_to_utc(x["end_time_cst"]), axis=1
        )
        manoeuvre['median_timestamp'] = manoeuvre.apply(
            lambda x: calculate_median(
                x['start_timestamp'], x['end_timestamp']
            ), axis=1
        )

        # Sort the DataFrame by end_timestamp in ascending order
        manoeuvre = manoeuvre.sort_values(by='end_timestamp', ascending=True)

        # Keep only the timestamps for ground truth
        self.proc_manoeuvre = manoeuvre[["start_timestamp", "end_timestamp", "median_timestamp"]].copy()
        return self.proc_manoeuvre

    def save_cleaned_file(self):
        self.proc_manoeuvre.to_csv(self.dst_file_path, index=True)


# Convert Year + DOY + Time to a standard datetime format
def convert_to_datetime(year, doy, hour, minute, second):
    """ Converts Year + Day-of-Year + Hour + Minute to a datetime object. """
    if pd.isna(year) or pd.isna(doy):
        return None
    year = int(float(year))
    doy = int(float(doy))
    date = dt.strptime(f"{year}-{doy}", "%Y-%j")  # Convert Year-DOY to YYYY-MM-DD

    if pd.isna(second):
        second = 0
    else:
        second = float(second)
    second_int = int(second) # Extracts whole seconds
    millisecond = int((second - second_int) * 1000) # Converts fraction to milliseconds
    return date.replace(hour=int(float(hour)), minute=int(float(minute)),
                        second=second_int, microsecond=millisecond * 1000)

def calculate_median(start_timestamp, end_timestamp):
    return start_timestamp + (end_timestamp - start_timestamp) / 2
