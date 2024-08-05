"""Usage: python visu/show_inputs.py +sid=<SID>.
Given a storm id of the form YYYYBBNN where BB is the basin code (e.g. AL),
NN is the storm number (e.g. 01), and YYYY is the year, this script will
display the inputs to the model for that storm.
"""

import matplotlib.pyplot as plt
import numpy as np
import hydra
from datetime import datetime as dt
from pathlib import Path
from omegaconf import OmegaConf


# List of satellite_sensor.SWATH values to display
SAT_SENSOR_PAIRS = [
    "GMI_GPM.KuGMI"
]


@hydra.main(config_path="../conf/", config_name="preproc", version_base=None)
def main(cfg):
    cfg = OmegaConf.to_container(cfg, resolve=True)
    inputs_path = Path(cfg['paths']['preprocessed_dataset'])

    sid = cfg['sid']
    year, basin = sid[:4], sid[4:6]
    inputs_path = inputs_path / year / basin / sid

    # inputs_path contains .npy files. Each file is an array of shape
    # (variables, lat, lon). Each file is named
    # YYYYmmDDHHMMSS-tc_primed.microwave.SEN_SAT.SWATH.npy
    # - Select only the files that are in SAT_SENSOR_PAIRS
    files = list(inputs_path.glob("*.npy"))
    files = [f for f in files if any(ssp in f.name for ssp in SAT_SENSOR_PAIRS)]
    # Retrieve the time steps from the file names
    time_steps = [dt.strptime(f.name[:14], "%Y%m%d%H%M%S") for f in files]
    # Retrieve the sensor and swath from the file names
    sensors = [f.name.split(".")[2] for f in files]
    swaths = [f.name.split(".")[3] for f in files]
    sensors_swaths = [f"{sensors[i]}.{swaths[i]}" for i in range(len(files))]
    # Create a map {time_step: {sensor_swath: data}}
    data = {}
    for i, (f, ts, ss) in enumerate(zip(files, time_steps, sensors_swaths)):
        if ts not in data:
            data[ts] = {}
        data[ts][ss] = np.load(f)

    # Display the data: one row per time step, one column per sensor_swath
    n_time_steps = len(data)
    n_sensors_swaths = len(SAT_SENSOR_PAIRS)
    fig, axs = plt.subplots(
        n_time_steps,
        n_sensors_swaths,
        figsize=(5 * n_sensors_swaths, 3 * n_time_steps),
        squeeze=False,
    )
    # Reduce the space between subplots
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    for i, (ts, d) in enumerate(data.items()):
        for j, ssp in enumerate(SAT_SENSOR_PAIRS):
            # If the sensor_swath is not in the data, display a white image
            if ssp in d:
                # Plot the last channel
                axs[i, j].imshow(d[ssp][-1], cmap="seismic")
                # The first two channels are the latitude and longitude of each pixel.
                # Use them as y-axis and x-axis labels respectively.
                axs[i, j].set_yticks([0, d[ssp].shape[1] - 1])
                axs[i, j].set_yticklabels(
                    [f"{d[ssp][0, 0, 0]:.2f}", f"{d[ssp][0, -1, 0]:.2f}"]  # latitude
                )
                axs[i, j].set_xticks([0, d[ssp].shape[2] - 1])
                axs[i, j].set_xticklabels(
                    [f"{d[ssp][1, 0, 0]:.2f}", f"{d[ssp][1, 0, -1]:.2f}"]  # longitude
                )
    # Write the titles: on the left, the time steps; on the top, the sensor_swaths
    for i, ts in enumerate(data.keys()):
        axs[i, 0].set_title(ts.strftime("%Y-%m-%d %H:%M:%S"), fontsize=7)
    for j, ssp in enumerate(SAT_SENSOR_PAIRS):
        axs[0, j].set_title(ssp, fontsize=7)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
