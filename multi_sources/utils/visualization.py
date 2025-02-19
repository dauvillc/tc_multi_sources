import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import numpy as np


def display_solution_html(batch, sol, time_grid, target_source):
    """Display the solution and groundtruth of the flow matching process using plotly.

    Args:
        batch (dict): The input batch containing the original data
        sol (torch.Tensor): Solution tensor of shape (T, B, C, H, W)
        time_grid (torch.Tensor): Time points of shape (T,)
        target_source (str): Name of the source being reconstructed

    Returns:
        plotly.graph_objects.Figure: Interactive figure with slider
    """
    # Convert tensors to numpy arrays
    sol_np = sol.detach().cpu().numpy()
    time_grid_np = time_grid.detach().cpu().numpy()
    groundtruth_np = batch[target_source]['values'][0].detach().cpu().numpy()  # Shape: (C, H, W)

    # Get the first batch item
    sol_np = sol_np[:, 0]  # Shape: (T, C, H, W)

    # Create figure with 2 rows
    fig = make_subplots(
        rows=2,
        cols=sol_np.shape[1],
        subplot_titles=[f"GT Channel {i+1}" for i in range(sol_np.shape[1])] +
                      [f"Solution Channel {i+1}" for i in range(sol_np.shape[1])],
    )

    # Create frames for the slider
    frames = []
    for t_idx, t in enumerate(time_grid_np):
        frame_data = []
        # First add groundtruth (constant across time)
        for channel in range(sol_np.shape[1]):
            frame_data.append(
                go.Heatmap(z=groundtruth_np[channel], showscale=True if t_idx == 0 else False)
            )
        # Then add solution
        for channel in range(sol_np.shape[1]):
            frame_data.append(
                go.Heatmap(z=sol_np[t_idx, channel], showscale=True if t_idx == 0 else False)
            )
        frames.append(go.Frame(data=frame_data, name=f"t={t:.2f}"))

    # Add initial data
    # First row: groundtruth
    for channel in range(sol_np.shape[1]):
        fig.add_trace(
            go.Heatmap(z=groundtruth_np[channel], showscale=True),
            row=1,
            col=channel + 1
        )
    # Second row: solution
    for channel in range(sol_np.shape[1]):
        fig.add_trace(
            go.Heatmap(z=sol_np[0, channel], showscale=True),
            row=2,
            col=channel + 1
        )

    # Update layout with slider and equal aspect ratio
    fig.update_layout(
        title=f"Flow Matching Evolution - {target_source}",
        height=800,  # Increase height to accommodate two rows
        sliders=[
            {
                "active": 0,
                "currentvalue": {"prefix": "Time: "},
                "pad": {"t": 50},
                "steps": [
                    {
                        "args": [
                            [f"t={t:.2f}"],
                            {
                                "frame": {"duration": 0, "redraw": True},
                                "mode": "immediate",
                            },
                        ],
                        "label": f"{t:.2f}",
                        "method": "animate",
                    }
                    for t in time_grid_np
                ],
            }
        ],
        updatemenus=[
            {
                "buttons": [
                    {
                        "args": [None, {"frame": {"duration": 500, "redraw": True}}],
                        "label": "Play",
                        "method": "animate",
                    },
                    {
                        "args": [[None], {"frame": {"duration": 0, "redraw": True}}],
                        "label": "Pause",
                        "method": "animate",
                    },
                ],
                "type": "buttons",
            }
        ],
    )

    # Update all subplot axes to have equal aspect ratio
    for i in range(1, 3):  # For both rows
        for j in range(1, sol_np.shape[1] + 1):  # For each channel
            fig.update_xaxes(scaleanchor=f"y{i}", scaleratio=1, row=i, col=j)
            fig.update_yaxes(scaleanchor=f"x{i}", scaleratio=1, row=i, col=j)

    fig.frames = frames

    return fig