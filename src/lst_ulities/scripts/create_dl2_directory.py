import argparse
import os
from collections import Counter
from pathlib import Path

import pandas as pd

from ..helper import find_lst_data_path
from ..run_statistics import RunStatistics


def main():
    parser = argparse.ArgumentParser(description="Create a DL2 directory structure.")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input data check files")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output directory for DL3 files")
    args = parser.parse_args()

    # Create the DL3 directory structure
    input_file = args.input
    df = pd.read_hdf(input_file, key="good_run_statistics")
    run_stat = RunStatistics(df=pd.DataFrame(df))

    run_number_list = run_stat.run_numbers
    date = run_stat["date"].astype("int")

    file_count = Counter()

    for irun, idate in zip(run_number_list, date):
        dl2_file_path: str = find_lst_data_path(idate, irun, level="dl2")
        if not dl2_file_path:
            print(f"No DL2 file found for run {irun}, date {idate}. Skipping.")
            continue
        output_dir = Path(args.output)
        os.makedirs(output_dir, exist_ok=True)
        try:
            os.symlink(dl2_file_path, output_dir / Path(dl2_file_path).name)
        except OSError as e:
            print(f"Error creating symlink for run {irun}, date {idate}, file {dl2_file_path}")
            print(f"  Error details: {e}")
            print(f"  Output directory: {output_dir}")
            print(f"  Skipping this file.")
            continue

    # Print summary: how many files in each output directory
    print("Summary of files per output directory:")
