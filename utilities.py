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
        for i, (d, l) in enumerate(zip(data, labels)):
            hf.create_dataset(f"data_{i}", data=d)
            hf.create_dataset(f"label_{i}", data=np.string_(l))


def load_data_from_hdf5(label, filename):
    """
    Load data from an HDF5 file based on the label.

    Parameters:
        label (str): Label of the data to load.
        filename (str): Name of the HDF5 file containing the data.

    Returns:
        numpy.ndarray: Array corresponding to the specified label.
    """
    with h5py.File(filename, "r") as hf:
        # Check if the label exists in the file
        if f"label_{label}" not in hf:
            raise ValueError(f"Label '{label}' not found in the HDF5 file.")

        # Load the dataset corresponding to the label
        data = hf[f"data_{label}"][:]
    
    return data


