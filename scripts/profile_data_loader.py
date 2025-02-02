import argparse
import logging
import os
from neural_mesh_simplification.data.data_profiler import DataLoaderProfiler


def main():
    parser = argparse.ArgumentParser(description="Profile the data loader.")
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="The directory where the dataset is stored.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        required=False,
        default=32,
        help="The batch size to use.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        required=False,
        default=os.cpu_count(),
        help="The number of worker processes to use.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Whether to shuffle the data.",
    )
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(level=logging.DEBUG)
    # Set trimesh logger to INFO level to suppress debug messages
    logging.getLogger("trimesh").setLevel(logging.INFO)

    profiler = DataLoaderProfiler(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=args.shuffle,
    )
    summary = profiler.profile()
    profiler.log_summary(summary)


if __name__ == "__main__":
    main()
