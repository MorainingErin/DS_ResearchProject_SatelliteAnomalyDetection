#!/bin/bash

# Activate the Conda environment
# source /c/ProgramData/anaconda3/etc/profile.d/conda.sh  # Adjust this path if Conda is installed elsewhere
# conda activate dsrp

# Run preprocessing
# python main.py --mode preprocess

# List of satellite names
satellites=("Fengyun-2F" "Fengyun-2H" "Sentinel-3A" "CryoSat-2" "SARAL")

# Loop through each satellite name
for satellite in "${satellites[@]}"; do
    echo "----------------------------------------"
    echo "Processing satellite: $satellite"
    
    # Run the Python program with the current satellite name
    # python main.py --mode analysis --satellite_name "$satellite" --verbose
    python main.py --mode train --satellite_name "$satellite" --verbose
    # python main.py --mode test --satellite_name "$satellite" --verbose
    
    echo "Finished processing satellite: $satellite"
    echo "----------------------------------------"
done