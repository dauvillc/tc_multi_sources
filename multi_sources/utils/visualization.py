import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from string import Template


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
        name: data for name, data in batch.items() 
        if data['avail'][sample_index].item() != -1  # Use sample_index
    }
    n_sources = len(available_sources)

    subplot_titles = []
    for source in available_sources.keys():
        dt = batch[source]['dt'][sample_index].item()  # Use sample_index
        subplot_titles.append(f"{source} (dt={dt:.3f}) Prediction")
        subplot_titles.append(f"{source} (dt={dt:.3f}) Ground Truth")
    
    # Create figure with subplots
    fig = make_subplots(
        rows=n_sources, 
        cols=2,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.05,
        vertical_spacing=0.05
    )
    
    # For each source, create a frame for each timestep
    frames = []
    for t in range(T):
        frame_data = []
        for i, (source_name, source_data) in enumerate(available_sources.items(), start=1):
            # Get prediction and ground truth
            pred = sol[source_name][t, sample_index, 0].detach().cpu().numpy()  # Use sample_index
            true = source_data['values'][sample_index, 0].detach().cpu().numpy()  # Use sample_index
            
            # For 2D sources
            if len(pred.shape) == 2:
                frame_data.append(go.Heatmap(
                    z=pred,
                    showscale=False,
                    colorscale='viridis',
                    xaxis=f"x{2*i-1}",
                    yaxis=f"y{2*i-1}"
                ))
                frame_data.append(go.Heatmap(
                    z=true,
                    showscale=False,
                    colorscale='viridis',
                    xaxis=f"x{2*i}",
                    yaxis=f"y{2*i}"
                ))
            # For 0D sources
            else:
                frame_data.append(go.Bar(
                    y=pred,
                    showlegend=False,
                    xaxis=f"x{2*i-1}",
                    yaxis=f"y{2*i-1}"
                ))
                frame_data.append(go.Bar(
                    y=true,
                    showlegend=False,
                    xaxis=f"x{2*i}",
                    yaxis=f"y{2*i}"
                ))
        
        frames.append(go.Frame(
            data=frame_data,
            name=f't{t}'
        ))
    
    # Add the initial data to the figure
    for i, (source_name, source_data) in enumerate(available_sources.items(), start=1):
        pred_init = sol[source_name][0, sample_index, 0].detach().cpu().numpy()  # Use sample_index
        true = source_data['values'][sample_index, 0].detach().cpu().numpy()  # Use sample_index
        
        if len(pred_init.shape) == 2:
            fig.add_trace(
                go.Heatmap(z=pred_init, showscale=False, colorscale='viridis'),
                row=i, col=1
            )
            fig.add_trace(
                go.Heatmap(z=true, showscale=False, colorscale='viridis'),
                row=i, col=2
            )
            # Configure axes for prediction subplot
            fig.update_xaxes(scaleanchor=f"y{2*i-1}", scaleratio=1, row=i, col=1)
            fig.update_yaxes(scaleanchor=f"x{2*i-1}", scaleratio=1, row=i, col=1)
            # Configure axes for ground truth subplot
            fig.update_xaxes(scaleanchor=f"y{2*i}", scaleratio=1, row=i, col=2)
            fig.update_yaxes(scaleanchor=f"x{2*i}", scaleratio=1, row=i, col=2)
        else:
            fig.add_trace(
                go.Bar(y=pred_init, showlegend=False),
                row=i, col=1
            )
            fig.add_trace(
                go.Bar(y=true, showlegend=False),
                row=i, col=2
            )
    
    # Update layout with slider
    fig.update_layout(
        sliders=[{
            'currentvalue': {'prefix': 't = '},
            'steps': [
                {
                    'args': [[f't{t}'], {
                        'frame': {'duration': 0, 'redraw': True},
                        'mode': 'immediate',
                    }],
                    'label': f'{time_grid[t]:.2f}',
                    'method': 'animate'
                } for t in range(T)
            ]
        }],
        updatemenus=[{
            'type': 'buttons',
            'showactive': False,
            'buttons': [{
                'label': 'Play',
                'method': 'animate',
                'args': [None, {
                    'frame': {'duration': 100, 'redraw': True},
                    'fromcurrent': True,
                    'mode': 'immediate',
                    'transition': {'duration': 0}
                }]
            }]
        }]
    )
    
    # Add frames to the figure
    fig.frames = frames
    
    return fig


def display_batch(batch, save_dir):
    """Creates and saves a figure for each sample in the batch.
    
    Args:
        batch (dict): Dict mapping source names to data dicts containing:
            - values (torch.Tensor): Values tensor of shape (B, C, ...)
            - avail (torch.Tensor): Availability flag of shape (B,)
        save_dir (str or Path): Directory where to save the figures
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Get batch size from first source
    first_source = next(iter(batch.values()))
    batch_size = first_source['values'].shape[0]
    n_sources = len(batch)
    
    # Create a figure for each sample in the batch
    for b in range(batch_size):
        # Calculate number of rows and columns for subplots
        n_rows = int(np.ceil(np.sqrt(n_sources)))
        n_cols = int(np.ceil(n_sources / n_rows))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
        axes = axes.flatten() if n_sources > 1 else [axes]
        
        for ax, (source_name, source_data) in zip(axes, batch.items()):
            values = source_data['values'][b].detach().cpu().numpy()  # (C, ...)
            avail = source_data['avail'][b].item()
            
            # For 2D sources
            if len(values.shape) == 3:
                # If multi-channel, only show first channel
                im = ax.imshow(values[0], cmap='viridis')
                plt.colorbar(im, ax=ax)
            # For 0D sources
            elif len(values.shape) == 1:
                ax.bar(range(len(values)), values)
            
            title = f"{source_name}\n"
            title += "Available" if avail == 1 else "Missing" if avail == -1 else "Masked"
            ax.set_title(title)
        
        # Remove any unused subplots
        for ax in axes[n_sources:]:
            ax.remove()
            
        plt.tight_layout()
        plt.savefig(save_dir / f"sample_{b}.png")
        plt.close()