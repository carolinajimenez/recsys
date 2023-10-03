"""
Save files
"""

import os
import pickle
import base64


def save_file_pkl(file, save_path, filename):
    """
    Save pickle file
    """
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, filename), "wb") as file_buffer:
        pickle.dump(file, file_buffer)


def read_file_pkl(save_path, filename):
    """
    Read pickle file
    """
    with open(os.path.join(save_path, filename), "rb") as file_buffer:
        loaded_object = pickle.load(file_buffer)
        return loaded_object