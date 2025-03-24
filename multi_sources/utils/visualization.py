import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from string import Template

from multi_sources.data_processing.grid_functions import crop_nan_border


def display_solution_html(batch, sol, time_grid, sample_index=0):
    """Display the solution and groundtruth of the flow matching process using plotly.

    Args:
        batch (dict): The input batch containing the original data
        sol (dict of str to torch.Tensor): The solution of the flow matching process,
            as a dict mapping source names to tensors of shape (T, B, C, ...).
        time_grid (torch.Tensor): Time points of shape (T,)
        sample_index (int, optional): Index of the sample to display. Defaults to 0.

    Returns:
        plotly.graph_objects.Figure: Interactive figure with slider
    """
    # Number of time steps
    T = len(time_grid)

    # Filter out unavailable sources
    available_sources = {
        name: data
        for name, data in batch.items()
        if data["avail"][sample_index].item() != -1  # Use sample_index
    }
    n_sources = len(available_sources)

    subplot_titles = []
    for source in available_sources.keys():
        dt = batch[source]["dt"][sample_index].item()  # Use sample_index
        subplot_titles.append(f"{source} (dt={dt:.3f}) Prediction")
        subplot_titles.append(f"{source} (dt={dt:.3f}) Ground Truth")

    # Create figure with subplots
    fig = make_subplots(
        rows=n_sources,
        cols=2,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.05,
        vertical_spacing=0.05,
    )

    # For each source, create a frame for each timestep
    frames = []
    for t in range(T):
        frame_data = []
        for i, (source_name, source_data) in enumerate(available_sources.items(), start=1):
            # Get prediction and ground truth
            pred = sol[source_name][t, sample_index, 0].detach().cpu().numpy()  # Use sample_index
            true = (
                source_data["values"][sample_index, 0].detach().cpu().numpy()
            )  # Use sample_index

            # For 2D sources
            if len(pred.shape) == 2:
                frame_data.append(
                    go.Heatmap(
                        z=pred,
                        showscale=False,
                        colorscale="viridis",
                        xaxis=f"x{2*i-1}",
                        yaxis=f"y{2*i-1}",
                    )
                )
                frame_data.append(
                    go.Heatmap(
                        z=true,
                        showscale=False,
                        colorscale="viridis",
                        xaxis=f"x{2*i}",
                        yaxis=f"y{2*i}",
                    )
                )
            # For 0D sources
            else:
                frame_data.append(
                    go.Bar(y=pred, showlegend=False, xaxis=f"x{2*i-1}", yaxis=f"y{2*i-1}")
                )
                frame_data.append(
                    go.Bar(y=true, showlegend=False, xaxis=f"x{2*i}", yaxis=f"y{2*i}")
                )

        frames.append(go.Frame(data=frame_data, name=f"t{t}"))

    # Add the initial data to the figure
    for i, (source_name, source_data) in enumerate(available_sources.items(), start=1):
        pred_init = sol[source_name][0, sample_index, 0].detach().cpu().numpy()  # Use sample_index
        true = source_data["values"][sample_index, 0].detach().cpu().numpy()  # Use sample_index

        if len(pred_init.shape) == 2:
            fig.add_trace(
                go.Heatmap(z=pred_init, showscale=False, colorscale="viridis"), row=i, col=1
            )
            fig.add_trace(go.Heatmap(z=true, showscale=False, colorscale="viridis"), row=i, col=2)
            # Configure axes for prediction subplot
            fig.update_xaxes(scaleanchor=f"y{2*i-1}", scaleratio=1, row=i, col=1)
            fig.update_yaxes(scaleanchor=f"x{2*i-1}", scaleratio=1, row=i, col=1)
            # Configure axes for ground truth subplot
            fig.update_xaxes(scaleanchor=f"y{2*i}", scaleratio=1, row=i, col=2)
            fig.update_yaxes(scaleanchor=f"x{2*i}", scaleratio=1, row=i, col=2)
        else:
            fig.add_trace(go.Bar(y=pred_init, showlegend=False), row=i, col=1)
            fig.add_trace(go.Bar(y=true, showlegend=False), row=i, col=2)

    # Update layout with slider
    fig.update_layout(
        sliders=[
            {
                "currentvalue": {"prefix": "t = "},
                "steps": [
                    {
                        "args": [
                            [f"t{t}"],
                            {
                                "frame": {"duration": 0, "redraw": True},
                                "mode": "immediate",
                            },
                        ],
                        "label": f"{time_grid[t]:.2f}",
                        "method": "animate",
                    }
                    for t in range(T)
                ],
            }
        ],
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 100, "redraw": True},
                                "fromcurrent": True,
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                    }
                ],
            }
        ],
    )

    # Add frames to the figure
    fig.frames = frames

    return fig


def display_realizations(sol, batch, avail_flags, save_filepath_prefix, deterministic=False,
                         display_fraction=1.0):
    """Given multiple solutions of the flow matching process, creates one figure
    per sample to display the solutions and groundtruth.
    Args:
        sol (dict of torch.Tensor): The solution of the flow matching process,
            as a dict mapping source names to tensors of shape (Np, B, C, ...)
            where Np is the number of realizations, B is the batch size, and C
            is the number of channels.
        batch (dict of dict of torch.Tensor): Dict mapping source names to data dicts.
            For each source, batch[source] must contains the following entries: "values",
            of shape (B, C, ...).
        avail_flags (dict): Dictionary {source: avail_flag_s} where avail_flag_s is a
            tensor of shape (B,) containing 1 if the value is available, 0 if it was
            available and masked, and -1 if it was not available.
        save_filepath_prefix (str or Path): Prefix of the filepath where the figure will be saved.
            The figure will be saved as save_filepath_prefix + "_{sample_idx}.png".
        deterministic (bool, optional): If True, expects sol to be a tensor of shape (B, C, ...).
        display_fraction (float, optional): Fraction of the samples to display. Defaults to 1.0.
    """
    save_filepath_prefix = Path(save_filepath_prefix)
    # Make sure parent directory exists
    save_filepath_prefix.parent.mkdir(parents=True, exist_ok=True)

    # Make the deterministic case compatible with the non-deterministic case
    if deterministic:
        sol = {src: sol_s.unsqueeze(0) for src, sol_s in sol.items()}

    # Extract batch size and number of realizations
    any_source = next(iter(sol.keys()))
    n_realizations = sol[any_source].shape[0]
    batch_size = sol[any_source].shape[1]

    # Take a fraction of the samples, evenly spaced
    samples_to_display = np.linspace(0, batch_size - 1, int(display_fraction * batch_size)).astype(int)

    # For each sample
    for sample_idx in samples_to_display:
        # Get available sources for this sample (either masked or available)
        sources = [
            source
            for source, flags in avail_flags.items()
            if flags[sample_idx].item() != -1  # Either masked (0) or available (1)
        ]

        if not sources:
            continue  # Skip if no sources are available or masked

        # Create a figure with n_realizations + 1 columns (realizations + groundtruth)
        # and one row per source
        fig, axs = plt.subplots(
            nrows=len(sources),
            ncols=n_realizations + 1,
            figsize=(3 * (n_realizations + 1), 3 * len(sources)),
            squeeze=False,
        )

        # For each source
        for src_idx, source in enumerate(sources):
            is_masked = avail_flags[source][sample_idx].item() == 0

            # For each realization
            for r_idx in range(n_realizations):
                ax = axs[src_idx, r_idx]
                # Get original coordinates
                coords = batch[source]["coords"][sample_idx]  # (2, H, W), lat/lon

                # Only show prediction if the source was masked
                if is_masked:
                    # Get prediction data
                    pred = sol[source][r_idx, sample_idx, 0]

                    # Display data based on dimensionality
                    if len(pred.shape) == 2:
                        # The images' borders may be NaN due to the batching system.
                        # Crop the NaN borders to display the images correctly.
                        pred = crop_nan_border(coords, [pred.unsqueeze(0)])[0].squeeze(0)
                        pred = pred.detach().cpu().numpy()
                        im = ax.imshow(pred, cmap="viridis")
                        # Add coords as axis labels
                        h, w = pred.shape
                        x_vals = coords[1, 0, :w]
                        y_vals = coords[0, :h, 0]
                        step_x = max(1, w // 5)
                        step_y = max(1, h // 5)
                        ax.set_xticks(range(0, w, step_x))
                        ax.set_yticks(range(0, h, step_y))
                        ax.set_xticklabels(
                            [f"{val:.2f}" for val in x_vals[0::step_x]], rotation=45
                        )
                        ax.set_yticklabels([f"{val:.2f}" for val in y_vals[0::step_y]])
                    else:  # For 0D or 1D data
                        pred = pred[0].detach().cpu().numpy()
                        ax.bar(range(len(pred)), pred)
                        (
                            ax.set_ylim([0, 1.2 * pred.max()])
                            if pred.max() > 0
                            else ax.set_ylim([1.2 * pred.min(), 0])
                        )

                    ax.set_title(f"Prediction {r_idx+1}")
                else:
                    ax.set_title(f"Not masked")
                    ax.axis("off")

            # Display groundtruth in the last column
            ax = axs[src_idx, -1]
            true = batch[source]["values"][sample_idx, 0]

            if len(true.shape) == 2:
                true = crop_nan_border(coords, [true.unsqueeze(0)])[0].squeeze(0)
                true = true.detach().cpu().numpy()
                im = ax.imshow(true, cmap="viridis")
                h, w = true.shape
                x_vals = coords[1, 0, :w]
                y_vals = coords[0, :h, 0]
                step_x = max(1, w // 5)
                step_y = max(1, h // 5)
                ax.set_xticks(range(0, w, step_x))
                ax.set_yticks(range(0, h, step_y))
                ax.set_xticklabels([f"{val:.2f}" for val in x_vals[0::step_x]], rotation=45)
                ax.set_yticklabels([f"{val:.2f}" for val in y_vals[0::step_y]])
            else:  # For 0D or 1D data
                true = true[0].detach().cpu().numpy()
                ax.bar(range(len(true)), true)
                (
                    ax.set_ylim([0, 1.2 * true.max()])
                    if true.max() > 0
                    else ax.set_ylim([1.2 * true.min(), 0])
                )

            ax.set_title(f"Ground Truth")

        plt.tight_layout()
        # Save figure
        save_path = f"{save_filepath_prefix}_{sample_idx}.png"
        plt.savefig(save_path)
        plt.close()
