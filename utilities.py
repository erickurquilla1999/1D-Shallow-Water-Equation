import h5py
import numpy as np

def save_data_to_hdf5(data, labels, filename):
    """
    Save data and labels to an HDF5 file.

    Parameters:
        data (list): List of numpy arrays containing data.
        labels (list): List of labels corresponding to the data arrays.
        filename (str): Name of the HDF5 file to save the data.

    Returns:
        None
    """
    # Check if the number of data arrays and labels match
    if len(data) != len(labels):
        raise ValueError("Number of data arrays and labels must be the same.")

    # Open HDF5 file for writing
    with h5py.File(filename, "w") as hf:
        # Save data and labels
        for label, d in zip(labels, data):
            hf.create_dataset(label, data=d)

def load_data_from_hdf5(dataset_name, filename):
    """
    Load data from an HDF5 file based on the provided dataset name.

    Parameters:
        dataset_name (str): Name of the dataset to load.
        filename (str): Name of the HDF5 file containing the data.

    Returns:
        numpy.ndarray: Array corresponding to the specified dataset name.
    """
    with h5py.File(filename, "r") as hf:
        # Check if the dataset exists in the file
        if dataset_name not in hf:
            raise ValueError(f"Dataset '{dataset_name}' not found in the HDF5 file '{filename}'.")

        # Load the dataset
        data = hf[dataset_name][()]
        
        # Check the shape of the dataset
        if data.shape == (1,):
            # If it's a single value, return the scalar value
            return data.item()
        else:
            # Otherwise, return the entire dataset
            return data