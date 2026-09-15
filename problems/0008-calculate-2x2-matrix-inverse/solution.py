import torch

def inverse_2x2(matrix) -> torch.Tensor | None:
    """
    Compute the inverse of a 2x2 matrix using PyTorch.
    
    Args:
        matrix: A 2x2 matrix (can be list, numpy array, or torch.Tensor)
    
    Returns:
        A 2x2 tensor containing the inverse, or None if the matrix is singular
    """
    m = torch.as_tensor(matrix, dtype=torch.float)
    # Your code here
    det = m[0][0]*m[1][1]-m[0][1]*m[1][0]
    if det == 0:
        return None
    else:
        A_inv = 1/det * torch.tensor([
            [m[1][1], -m[0][1]],
        [-m[1][0], m[0][0]]
        ])
        return A_inv