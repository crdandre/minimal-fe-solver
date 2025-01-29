import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import cg
from scipy.sparse.linalg import spsolve

def conjugate_gradient_solve(K: csr_matrix, f: np.ndarray, tol: float = 1e-14, max_iter: int = None) -> np.ndarray:
    """Solve Kx = f using scipy's conjugate gradient solver"""
    n = K.shape[0]
    
    # Set max_iter to matrix size if not specified
    if max_iter is None:
        max_iter = n
    
    # Scale the system to improve conditioning
    scale = 1.0 / np.sqrt(K.diagonal())
    K_scaled = K.multiply(scale[:, None]).multiply(scale[None, :])
    f_scaled = f * scale
    
    # Solve scaled system
    x_scaled, info = cg(K_scaled, f_scaled, atol=tol, maxiter=max_iter)
    
    # Unscale solution
    x = x_scaled * scale
    
    # Compute true residual
    residual = np.linalg.norm(K @ x - f) / np.linalg.norm(f)
    print(f"Relative residual (CG): {residual:.2e}")
    print(f"Iterations used: {info if info > 0 else max_iter}")
    
    return x
