# -*- coding: utf-8 -*-
#
# Define the tool that will be used for other program.
#
import os
import shutil
import json
import pickle
import pandas as pd


def read_txt(path):
    """read text file from path."""
    with open(path, "r") as f:
        return f.read().splitlines()


def read_json(path):
    """read json file from path."""
    with open(path, 'r') as f:
        return json.load(f)


def read_csv(path, delimiter="\t", header=-1):
    """read csv file from path."""
    return pd.read_csv(path, delimiter=delimiter, header=header)


def write_txt(data, out_path, type="w"):
    """write the data to the txt file."""
    with open(out_path, type) as f:
        f.write(data.encode("utf-8"))


def load_pickle(path):
    """load data by pickle."""
    with open(path, 'rb') as handle:
        return pickle.load(handle)


def write_pickle(data, path):
    """dump file to dir."""
    print("write --> data to path: {}\n".format(path))
    with open(path, 'wb') as handle:
        pickle.dump(data, handle)


def build_dir(path, force):
    """build directory."""
    if os.path.exists(path) and force:
        shutil.rmtree(path)
        os.mkdir(path)
    elif not os.path.exists(path):
        os.mkdir(path)
    return path
