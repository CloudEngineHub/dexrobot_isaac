import re
import pandas as pd
import numpy as np
from pathlib import Path


# Function to extract numbers from the string and convert them to a NumPy array
def parse_array(array_str):
    # Use regular expression to find all floating-point numbers, including scientific notation
    numbers = re.findall(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|\d+", array_str)

    # Convert the extracted numbers to float and create a NumPy array
    return np.array(list(map(float, numbers)))


def extract_pregrasp_from_csv(csv_path, tag, pos_offset=[0.0, 0.0, 1.2 - 0.85]):
    # Load csv file
    df = pd.read_csv(csv_path)
    # Extract joint positions and hand pose from last row
    pattern_joint = "r_f_joint\d_\d_pos"
    pattern_hand = "AR.._pos"
    pattern = f"({pattern_joint}|{pattern_hand})"
    data_entry = df.iloc[[-1]].filter(regex=pattern)
    data_entry["hand_pos_x"] = data_entry["ARTx_pos"] + pos_offset[0]
    data_entry["hand_pos_y"] = data_entry["ARTy_pos"] + pos_offset[1]
    data_entry["hand_pos_z"] = data_entry["ARTz_pos"] + pos_offset[2]
    data_entry["hand_euler_X"] = data_entry["ARRx_pos"]
    data_entry["hand_euler_Y"] = data_entry["ARRy_pos"]
    data_entry["hand_euler_Z"] = data_entry["ARRz_pos"]
    data_entry = data_entry.drop(columns=data_entry.filter(regex=pattern_hand).columns)
    data_entry.insert(0, "tag", [tag])
    return data_entry


def make_dataset(parent_dir, patter_str=None):
    """Make dataset from all csv files matching parent_dir/dir_matching_pattern_str/tag/recorded_data.csv"""
    base_path = Path(parent_dir)
    if patter_str is not None:
        pattern = re.compile(patter_str)
    else:
        pattern = re.compile("^.*")
    experiment_dirs = [
        subdir
        for subdir in base_path.iterdir()
        if subdir.is_dir() and pattern.match(subdir.name)
    ]
    dfs = []
    for experiment in experiment_dirs:
        for tag in experiment.iterdir():
            if tag.is_dir():
                csv_path = tag / "recorded_data.csv"
                if csv_path.exists():
                    data_entry = extract_pregrasp_from_csv(csv_path, tag.name)
                    dfs.append(data_entry)
    return pd.concat(dfs, ignore_index=True)


def filter_dataset(df, percentile=0.8):
    # Compute the average hand_pos_x, hand_pos_y, and hand_pos_z
    avg_x = df.filter(regex="hand_pos_x").mean(axis=1)
    avg_y = df.filter(regex="hand_pos_y").mean(axis=1)
    avg_z = df.filter(regex="hand_pos_z").mean(axis=1)

    # Compute distance from the average position
    distance = np.linalg.norm(
        df.filter(regex="hand_pos_.").values - np.array([avg_x, avg_y, avg_z]).T, axis=1
    )
    # Find the percentile of the distance
    threshold = np.percentile(distance, percentile * 100)
    # Eliminate the entries with distance greater than the threshold
    df_filtered = df[distance <= threshold]
    return df_filtered


dataset = make_dataset("original")
dataset = filter_dataset(dataset)
dataset.to_csv("dataset.csv", index=False)
