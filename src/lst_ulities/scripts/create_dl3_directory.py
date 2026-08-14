import argparse
import os
from collections import Counter
from pathlib import Path

import pandas as pd

from ..helper import find_lst_data_path
from ..run_statistics import RunStatistics


def main():
    parser = argparse.ArgumentParser(description="Create a DL3 directory structure.")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input data check files")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output directory for DL3 files")
    args = parser.parse_args()

    # Create the DL3 directory structure
    input_file = args.input
    df = pd.read_hdf(input_file, key="good_run_statistics")
    run_stat = RunStatistics(df=df)

    run_number_list = run_stat.run_numbers
    date = run_stat["date"].astype("int")

    file_count = Counter()

    for irun, idate in zip(run_number_list, date):
        dl3_file_path = find_lst_data_path(idate, irun, level="dl3")
        for dl3_file in dl3_file_path:
            upper_dir = Path(dl3_file).parent.name
            output_dir = Path(args.output) / upper_dir
            os.makedirs(output_dir, exist_ok=True)
            os.symlink(dl3_file, output_dir / Path(dl3_file).name)
            file_count[upper_dir] += 1

    # Print summary: how many files in each output directory
    print("Summary of files per output directory:")
    for directory, count in sorted(file_count.items()):
        print(f"  {directory}: {count} file(s)")
