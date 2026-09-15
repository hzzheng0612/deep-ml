import torch

def calculate_eigenvalues(matrix: torch.Tensor) -> torch.Tensor:
    """
    Compute eigenvalues of a 2x2 matrix using PyTorch.
    Input: 2x2 tensor; Output: 1-D tensor with the two eigenvalues in descending order (highest to lowest).
    """
    # Your implementation here
    trace = matrix[0][0]+matrix[1][1]
    det = matrix[0][0]*matrix[1][1]-matrix[0][1]*matrix[1][0]
    lambda_1 = (trace+(trace**2-4*det)**0.5)/2
    lambda_2 = (trace-(trace**2-4*det)**0.5)/2
    result = [lambda_1, lambda_2]
    return torch.tensor(result)
