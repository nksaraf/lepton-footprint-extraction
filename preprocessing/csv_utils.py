import os

import pandas as pd
from sklearn.model_selection import train_test_split


def save_to_csv(file_path, data, columns, **kwargs):
    df = pd.DataFrame(data=data, columns=columns)
    df = df.dropna()
    df.to_csv(file_path, **kwargs)


def combine_csvs(columns, file_names=[]):
    frame = pd.DataFrame(columns=columns)
    for filename in file_names:
        table = pd.read_csv(filename, header=0, names=columns, na_filter=False)
        frame = frame.append(table, ignore_index=True)
    return frame


def train_val_test_split(file_names, seed, train_filepath, val_filepath, test_filepath, part=1.0):
    columns = ("image_id", "file_path_image", "file_path_mask")
    frame = combine_csvs(columns, file_names)
    if part < 1.0:
        frame, _ = train_test_split(frame, test_size=1-part, random_state=seed)
    train_val, test = train_test_split(frame, test_size=0.1, random_state=seed)
    train, val = train_test_split(train_val, test_size=0.2, random_state=seed)
    save_to_csv(train_filepath, train, index=False, columns=columns)
    save_to_csv(val_filepath, val, index=False, columns=columns)
    save_to_csv(test_filepath, test, index=False, columns=columns)


def create_local_version(filename):
    columns = ("image_id", "file_path_image", "file_path_mask")
    table = pd.read_csv(filename, header=0, names=columns, na_filter=False)
    table['file_path_image'] = table['file_path_image'].map(lambda a: a[12:])
    table['file_path_mask'] = table['file_path_mask'].map(lambda a: a[12:])
    table.to_csv('{}.local.csv'.format(os.path.splitext(filename)[0]), index=False, columns=columns)

