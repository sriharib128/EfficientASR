import argparse
import glob
import os
import random
from joblib import Parallel, delayed
import soundfile
from random import sample
from tqdm import tqdm


def get_parser():
    """Get command-line arguments parser."""
    parser = argparse.ArgumentParser(description="Generate manifest files from audio files.")
    parser.add_argument(
        "root", metavar="DIR", help="Root directory containing audio files to index"
    )
    parser.add_argument(
        "--valid-percent",
        default=0.01,
        type=float,
        metavar="D",
        help="Percentage of data to use as validation set (between 0 and 1)",
    )
    parser.add_argument(
        "--dest", default=".", type=str, metavar="DIR", help="Output directory"
    )
    parser.add_argument(
        "--train-name", default="train", type=str, help="Name of the training TSV file"
    )
    parser.add_argument(
        "--valid-name", default="valid", type=str, help="Name of the validation TSV file"
    )
    parser.add_argument(
        "--jobs", default=-1, type=int, help="Number of jobs to run in parallel"
    )
    parser.add_argument(
        "--ext", default="flac", type=str, metavar="EXT", help="Extension to look for (e.g., flac, wav)"
    )
    parser.add_argument(
        "--seed", default=42, type=int, metavar="N", help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--path-must-contain",
        default=None,
        type=str,
        metavar="FRAG",
        help="If set, path must contain this substring for a file to be included",
    )
    return parser


def read_file(fname, args, dir_path):
    """Read and process each audio file to extract its path and frame count."""
    file_path = os.path.realpath(fname)

    if args.path_must_contain and args.path_must_contain not in file_path:
        return ""  # Skip files that don't match the given fragment

    # Ensure sample rate is 16000
    if soundfile.info(fname).samplerate != 16000:
        return ""  # Skip files that don't have the correct sample rate

    frames = soundfile.info(fname).frames
    # Skip files with invalid frame count (e.g., empty or too long)
    if 0 < frames <= 480000:
        return f"{os.path.relpath(file_path, dir_path)}\t{frames}\n"
    return ""


def main(args):
    """Main function to generate training and validation manifest files."""
    assert 0 <= args.valid_percent <= 1.0, "valid-percent must be between 0 and 1"

    dir_path = os.path.realpath(args.root)
    search_path = os.path.join(dir_path, "**/*." + args.ext)

    # Initialize the random seed
    random.seed(args.seed)

    # Collect all valid audio file data in parallel
    file_data = Parallel(n_jobs=args.jobs)(
        delayed(read_file)(fname, args, dir_path) for fname in tqdm(glob.iglob(search_path, recursive=True))
    )

    # Filter out empty strings (files that didn't pass the checks)
    file_data = [data for data in file_data if data]

    # Split into training and validation sets
    num_valid = int(len(file_data) * args.valid_percent)
    valid_samples = sample(file_data, num_valid)
    train_samples = list(set(file_data) - set(valid_samples))

    # Include the directory at the top of the manifest file
    train_samples.insert(0, str(dir_path).strip() + "\n")
    valid_samples.insert(0, str(dir_path).strip() + "\n")

    # Create output directory if it doesn't exist
    os.makedirs(args.dest, exist_ok=True)

    # Save training manifest file
    train_manifest_path = os.path.join(args.dest, args.train_name + ".tsv")
    with open(train_manifest_path, "w+") as train_f:
        train_f.writelines(train_samples)
        print(f"** Writing Training Manifest File done with {len(train_samples)} records")

    # Save validation manifest file if needed
    if valid_samples:
        valid_manifest_path = os.path.join(args.dest, args.valid_name + ".tsv")
        with open(valid_manifest_path, "w+") as valid_f:
            valid_f.writelines(valid_samples)
            print(f"** Writing Validation Manifest File done with {len(valid_samples)} records")

    print("## Manifest Creation Done")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
