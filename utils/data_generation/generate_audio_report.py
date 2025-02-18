import pandas as pd
import numpy as np
import argparse
from tabulate import tabulate

def generate_report(tsv_file):
    """
    Generates a statistical report on audio file durations from a TSV file.
    
    :param tsv_file: Path to the TSV file containing 'file' and 'frames' columns.
    """
    try:
        # Load data
        df = pd.read_csv(tsv_file, sep='\t', header=0, names=['file', 'frames'])
        
        if df.empty:
            print(f"⚠️ Warning: The file '{tsv_file}' is empty!")
            return
        
        # Compute durations
        df['duration_sec'] = df['frames'] / 16000

        # Basic statistics
        total_files = len(df)
        total_duration_hrs = df['duration_sec'].sum() / 3600
        min_duration = df['duration_sec'].min()
        max_duration = df['duration_sec'].max()
        mean_duration = df['duration_sec'].mean()

        # Count files with duration ≤ 1s and ≤ 5s
        files_le_1s = df.query("duration_sec <= 1")['duration_sec']
        files_le_5s = df.query("duration_sec <= 5")['duration_sec']

        # Create table data
        table = [
            ["Total Files", total_files],
            ["Total Duration (hrs)", f"{total_duration_hrs:.2f}"],
            ["Min Duration (sec)", f"{min_duration:.2f}"],
            ["Max Duration (sec)", f"{max_duration:.2f}"],
            ["Mean Duration (sec)", f"{mean_duration:.2f}"],
            ["Files ≤ 1s", f"{len(files_le_1s)} ({files_le_1s.sum()/3600:.2f} hrs)"],
            ["Files ≤ 5s", f"{len(files_le_5s)} ({files_le_5s.sum()/3600:.2f} hrs)"]
        ]

        # Print report with tabulate
        print("=" * 60)
        print(f"📊 Report for: {tsv_file}")
        print("=" * 60)
        print(tabulate(table, headers=["Metric", "Value"], tablefmt="grid"))
        print("=" * 60)

    except FileNotFoundError:
        print(f"❌ Error: File '{tsv_file}' not found!")
    except pd.errors.EmptyDataError:
        print(f"❌ Error: File '{tsv_file}' is empty!")
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze audio file durations from a TSV file.")
    parser.add_argument('--tsv', type=str, required=True, help="Path to the TSV file")
    args = parser.parse_args()

    generate_report(args.tsv)
