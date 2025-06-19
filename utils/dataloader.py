import pandas as pd


class DataLoader():
    def __init__(self, orbital_path, manoeuvre_path, test_size=0.2):
        self.test_size = test_size
        self.orbital_path = orbital_path
        self.manoeuvre_path = manoeuvre_path
        self.orbital_data = None
        self.train_data = None
        self.test_data = None
        self.manoeuvre_data = None
        self.satellite_name = orbital_path.stem
        self.readable_labels = {
            "Unnamed: 0": "timestamp",
            "eccentricity": "Eccentricity",
            "argument of perigee": "Argument of Perigee",
            "inclination": "Inclination",
            "mean anomaly": "Mean Anomaly",
            "Brouwer mean motion": "Brouwer Mean Motion",
            "right ascension": "RAAN"
        }
    
    def load_orbital_data(self):
        # Load the data from the satellite path
        data = pd.read_csv(self.orbital_path)
        
        data.rename(columns=self.readable_labels, inplace=True)
        data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
        data.dropna(subset=[data.columns[0]], inplace=True)

        n = len(data)
        test_split = int(n * (1 - self.test_size))

        self.orbital_data = data
        self.train_data = data.iloc[:test_split]
        self.test_data = data.iloc[test_split:]

    def get_train_test_data(self, element):
        return self.train_data[element], self.test_data[element]
    
    def get_all_data(self, element):
        return self.orbital_data[element]
    
    def get_elements(self):
        return self.train_data.columns[1:]
    
    def get_selected_elements(self):
        return [
            self.readable_labels[x]
            for x in ("Brouwer mean motion", ) 
            # for x in ("Brouwer mean motion", "eccentricity", "argument of perigee", "inclination", "right ascension", "mean anomaly")
        ]
    
    def get_all_timestamps(self):
        return self.orbital_data["timestamp"]
    
    def load_manoeuvre_data(self):
        # Load the manoeuvre data from the given path
        self.manoeuvre_data = pd.read_csv(self.manoeuvre_path, index_col=0)
        self.manoeuvre_data["end_timestamp"] = pd.to_datetime(self.manoeuvre_data["end_timestamp"], 
                                                              format='%Y-%m-%d %H:%M:%S', errors='coerce')
        # # For Fengyun-2F, the manoeuvre data is in the format DD/MM/YYYY HH:MM:SS
        # if self.satellite_name == "Fengyun-2F":
        #     self.manoeuvre_data["end_timestamp"] = pd.to_datetime(self.manoeuvre_data["end_timestamp"], 
        #                                                           format='%d/%m/%Y %H:%M', errors='coerce')
        # else:
        #     # For other satellites, the manoeuvre data is in the format YYYY-MM-DD HH:MM:SS
        #     self.manoeuvre_data["end_timestamp"] = pd.to_datetime(self.manoeuvre_data["end_timestamp"], 
        #                                                         format='%Y-%m-%d %H:%M:%S', errors='coerce')
    
    def get_manoeuvre_data(self):
        return self.manoeuvre_data