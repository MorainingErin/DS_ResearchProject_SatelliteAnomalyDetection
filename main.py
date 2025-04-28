from utils.config import get_args
from utils import exploratory_analysis, preprocess_dataset
from models.training import train
from models.testing import test


def main():

    args = get_args()

    if args.mode == "preprocess":
        preprocess_dataset(args.data_dir, args.verbose)

    elif args.mode == "analysis":
        # Perform exploratory analysis on the data
        exploratory_analysis(args.data_dir, args.output_dir, args.satellite_name, args.verbose)

    elif args.mode == "train":
        # Train the models
        train(args.data_dir, args.output_dir, args.satellite_name, args.verbose)
    
    else:
        test(args.data_dir, args.output_dir, args.satellite_name, args.verbose)


if __name__ == "__main__":
    main()