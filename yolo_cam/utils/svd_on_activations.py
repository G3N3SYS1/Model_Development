# import numpy as np
#
# def get_2d_projection(activation_batch):
#     # TBD: use pytorch batch svd implementation
#     activation_batch[np.isnan(activation_batch)] = 0
#     projections = []
#     for activations in activation_batch:
#         reshaped_activations = (activations).reshape(
#             activations.shape[0], -1).transpose()
#         # Centering before the SVD seems to be important here,
#         # Otherwise the image returned is negative
#         reshaped_activations = reshaped_activations - \
#             reshaped_activations.mean(axis=0)
#         U, S, VT = np.linalg.svd(reshaped_activations, full_matrices=True)
#         projection = reshaped_activations @ VT[0, :]
#         projection = projection.reshape(activations.shape[1:])
#         projections.append(projection)
#     return np.float32(projections)

from scipy.sparse.linalg import svds

def get_2d_projection(activation_batch):
    reshaped_activations = activation_batch.reshape(activation_batch.shape[0], -1)
    reshaped_activations -= reshaped_activations.mean(axis=0)
    U, S, VT = svds(reshaped_activations, k=1)  # Compute only the largest singular vector
    projection = reshaped_activations @ VT[0, :]
    return projection.reshape(activation_batch.shape[1:])
