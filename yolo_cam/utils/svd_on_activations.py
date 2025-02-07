import numpy as np
from scipy.sparse.linalg import svds


def get_2d_projection(activation_batch):
    """
    Computes 2D projection of activations using SVD.
    Handles edge cases and various activation shapes safely.

    Args:
        activation_batch: numpy array of shape (batch_size, channels, height, width)
                         or (channels, height, width)

    Returns:
        2D projection of the activations
    """
    # Ensure activation_batch is at least 3D
    if len(activation_batch.shape) < 3:
        activation_batch = activation_batch[np.newaxis, ...]

    # If we don't have batch dimension, add it
    if len(activation_batch.shape) == 3:
        activation_batch = activation_batch[np.newaxis, ...]

    # Reshape activations to 2D matrix
    b, c, h, w = activation_batch.shape
    reshaped_activations = activation_batch.reshape(b, -1)  # Shape: (b, c*h*w)

    # Center the data
    reshaped_activations = reshaped_activations - reshaped_activations.mean(axis=1, keepdims=True)

    # Determine valid k value for SVD
    min_dim = min(reshaped_activations.shape)
    k = min(min_dim - 1, 1)  # Use k=1 or smaller if limited by dimension

    try:
        # Compute SVD for the first singular vector
        if k > 0:
            U, S, VT = svds(reshaped_activations, k=k)
            projection = reshaped_activations @ VT.T
        else:
            # Fallback for very small activations
            projection = reshaped_activations

        # Reshape back to original spatial dimensions
        projection = projection.reshape(b, h, w)

        # Return the first batch item
        return projection[0]

    except Exception as e:
        print(f"SVD computation failed: {str(e)}")
        # Fallback: return mean across channels
        return np.mean(activation_batch[0], axis=0)