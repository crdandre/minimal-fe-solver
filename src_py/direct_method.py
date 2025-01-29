import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

def lu_solve(K: csr_matrix, f: np.ndarray) -> np.ndarray:
    """Solve Kx = f using sparse direct solver with improved numerical stability"""
    if not (K.shape[0] == K.shape[1]):
        raise ValueError("Matrix K must be square")
    
    # Convert to double precision if not already
    K = K.astype(np.float64)
    f = f.astype(np.float64)
    
    # Add small regularization to diagonal to improve conditioning
    epsilon = 1e-10 * abs(K.diagonal()).mean()
    K = K + csr_matrix((epsilon * np.ones(K.shape[0]), 
                       (range(K.shape[0]), range(K.shape[0]))), 
                       shape=K.shape)
    
    # Scale the system
    scale = 1.0 / np.sqrt(abs(K.diagonal()))
    K_scaled = K.multiply(scale[:, None]).multiply(scale[None, :])
    f_scaled = f * scale
    
    # Ensure symmetry
    K_scaled = 0.5 * (K_scaled + K_scaled.T)
    
    # Solve scaled system
    x_scaled = spsolve(K_scaled, f_scaled)
    
    # Unscale solution
    x = x_scaled * scale
    
    # Check solution quality
    residual = np.linalg.norm(K @ x - f) / np.linalg.norm(f)
    print(f"Relative residual (Direct): {residual:.2e}")
    
    return x
