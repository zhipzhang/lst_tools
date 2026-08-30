import argparse
import os
from collections import Counter
from pathlib import Path

from ..datacheck import RunStatistics
from ..helper import find_lst_data_path


def main():
    parser = argparse.ArgumentParser(description="Create a DL3 directory structure.")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input data check files")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output directory for DL3 files")
    args = parser.parse_args()

    # Create the DL3 directory structure
    input_file = args.input
    run_stat = RunStatistics.from_file(input_file)

    run_number_list = run_stat.run_numbers
    date = run_stat["date"].astype("int")

    file_count = Counter()

    for irun, idate in zip(run_number_list, date):
        dl3_file_path = find_lst_data_path(idate, irun, level="dl3")
        for dl3_file in dl3_file_path:
            split_level = dl3_file.split("/")
            upper_dir = "/".join(split_level[-6:-1])  # Get the directory name three levels up
            output_dir = Path(args.output) / upper_dir
            os.makedirs(output_dir, exist_ok=True)
            try:
                os.symlink(dl3_file, output_dir / Path(dl3_file).name)
                file_count[upper_dir] += 1
            except OSError as e:
                print(f"Error creating symlink for run {irun}, date {idate}, file {dl3_file}")
                print(f"  Error details: {e}")
                print(f"  Output directory: {output_dir}")
                print("  Skipping this file.")
                continue

    # Print summary: how many files in each output directory
    print("Summary of files per output directory:")
    for directory, count in sorted(file_count.items()):
        print(f"  {directory}: {count} file(s)")
