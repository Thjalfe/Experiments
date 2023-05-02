import os
import numpy as np
import matplotlib.pyplot as plt


def save_plot(filename, bbox_inches="tight", file_format="pdf", **kwargs):
    """
    Save the current plot with specified settings.

    Args:
        filename (str): The file name (including path) to save the plot.
        bbox_inches (str, optional): The bbox_inches setting for plt.savefig. Default is 'tight'.
        file_format (str, optional): The file format to save the plot. Default is 'pdf'.
        dpi (int, optional): The resolution in dots per inch. Default is 300.
        **kwargs: Additional keyword arguments for plt.savefig().
    """

    # Ensure the file format is correct

    if not filename.endswith(f".{file_format}"):
        filename += f".{file_format}"

    # Create the directory if it doesn't exist
    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(filename, format=file_format, bbox_inches=bbox_inches, **kwargs)


def plot_top_n_datasets(sorted_peak_data, datasets, n, pairs):
    top_n_indices = list(sorted_peak_data.keys())[:n]

    plt.figure()
    for index in top_n_indices:
        data = datasets[index]

        # plt.plot(data[:, 0], data[:, 1], marker="o", linestyle="-")
        plt.plot(
            data[:, 0],
            data[:, 1],
            label=f"CE: -{np.min(sorted_peak_data[index]['differences']):.2f} dB",
        )
        plt.title(f"Top {n} datasets based on CE for {pairs[0]} nm and {pairs[1]} nm")
        plt.xlabel("Wavelength [nm]")  # Replace with the appropriate label
        plt.ylabel("Measured power [dBm]")  # Replace with the appropriate label
        plt.grid(True)
        plt.legend()
