import argparse
import os
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser(description="Process some integers.")

    parser.add_argument("--mode", type=str, choices=["preprocess", "analysis", "train", "test", "evaluate"], 
                        default="analysis",
                        help="Mode of operation: preprocess, analysis, train or test")

    parser.add_argument("--data_dir", type=Path, default=Path("./satellite_data"),
                        help="Directory containing the satellite data")
    
    parser.add_argument("--output_dir", type=Path, default=Path("./output"),
                        help="Directory to save the processed data")
    
    parser.add_argument("--satellite_name", type=str, default="CryoSat-2",
                        help="Name of the satellite to process")
    
    # parser.add_argument("--element", type=str,
    #                     help="Name of the orbit element")

    parser.add_argument("--verbose", action='store_true', default=True,
                        help="Enable verbose output")
    
    return parser.parse_args()
