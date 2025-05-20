import torch
import matplotlib.pyplot as plt
import imageio.v3 as imageio
import io
import numpy as np

if __name__ == "__main__":
    weights = torch.load("a_tracked_ws.pt")
    output_filename="weight_evolution.gif"
    fps=10
    frames_to_show=100
  
    # Select a subset of frames to keep GIF manageable
    step = max(1, len(weights) // frames_to_show)
    selected_weights = weights[::step][:frames_to_show]
    
    # Compute global vmin and vmax from the first weight matrix
    global_max_abs = np.abs(selected_weights[0]).max()
    vmin, vmax = -global_max_abs, global_max_abs
    print(f"Using vmin={vmin:.2f}, vmax={vmax:.2f} for colormap scaling.")
    
    # List to store in-memory image buffers
    frames = []
    for idx, weight in enumerate(selected_weights):
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111)
        ax.imshow(weight, cmap='bwr', vmin=vmin, vmax=vmax, interpolation='nearest')
        ax.axis('off')
        
        # Save plot to in-memory buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        frame = imageio.imread(buf)
        frames.append(frame)
    
    imageio.imwrite(output_filename, frames, format='GIF', duration=1000/fps, loop=0)
    print(f"GIF saved as {output_filename}")


