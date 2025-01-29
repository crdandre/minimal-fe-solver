import numpy as np
from scipy.sparse import coo_matrix
from typing import Dict, List, Tuple

from .fem_types import MeshData, MaterialProperties, BoundaryCondition

class FEMSolver:
    def __init__(self, mesh: MeshData):
        self.mesh = mesh
        self.materials: Dict[int, MaterialProperties] = {}
        self.constraints: List[BoundaryCondition] = []
        self.prescribed: List[BoundaryCondition] = []
        self.K = None  # Global stiffness matrix
        self.u = None  # Solution vector
    
    def add_material(self, component_id: int, E: float, nu: float) -> None:
        """Add material properties for a component"""
        D = self._create_constitutive_matrix(E, nu)
        self.materials[component_id] = MaterialProperties(E=E, nu=nu, D=D)
    
    def add_constraint(self, node_id: int, dof: int) -> None:
        """Add fixed DOF constraint"""
        self.constraints.append(BoundaryCondition(node_id, dof, 0.0))
    
    def add_prescribed_displacement(self, node_id: int, dof: int, value: float) -> None:
        """Add prescribed displacement"""
        self.prescribed.append(BoundaryCondition(node_id, dof, value))
    
    def _create_constitutive_matrix(self, E: float, nu: float) -> np.ndarray:
        """Create the constitutive (D) matrix for linear elasticity"""
        D = np.zeros((6, 6))
        c = E / ((1 + nu) * (1 - 2 * nu))
        
        # Main diagonal terms for normal strains
        D[0:3, 0:3] = c * (1 - nu)
        
        # Shear terms need to be c * (1 - 2*nu)/2
        D[3:6, 3:6] = c * (1 - 2*nu) / 2
        
        # Off-diagonal terms for Poisson effect
        for i in range(3):
            for j in range(3):
                if i != j:
                    D[i, j] = c * nu
        return D
    
    def _compute_element_stiffness(self, 
                                 element_coords: np.ndarray, 
                                 D: np.ndarray) -> np.ndarray:
        """Compute element stiffness matrix"""
        K_e = np.zeros((24, 24))
        gauss_points = [-1/np.sqrt(3), 1/np.sqrt(3)]
        
        for xi in gauss_points:
            for eta in gauss_points:
                for zeta in gauss_points:
                    B = self._compute_B_matrix(xi, eta, zeta)
                    J = self._compute_jacobian(xi, eta, zeta, element_coords)
                    detJ = np.linalg.det(J)
                    
                    K_e += B.T @ D @ B * detJ * (1/np.sqrt(3))**3
        
        return K_e
    
    def assemble_system(self) -> None:
        """Assemble global stiffness matrix"""
        ndof = len(self.mesh.nodes) * 3
        rows, cols, data = [], [], []
        
        for elem_idx, elem in enumerate(self.mesh.elements):
            comp_id = elem.component_id
            D = self.materials[comp_id].D
            elem_coords = self.mesh.node_coords[self.mesh.element_connectivity[elem_idx]]
            
            K_e = self._compute_element_stiffness(elem_coords, D)
            dofs = np.array([[3*n, 3*n+1, 3*n+2] 
                            for n in self.mesh.element_connectivity[elem_idx]]).flatten()
            
            for i in range(24):
                for j in range(24):
                    rows.append(dofs[i])
                    cols.append(dofs[j])
                    data.append(K_e[i,j])
        
        self.K = coo_matrix((data, (rows, cols)), shape=(ndof, ndof)).tocsr()
    
    def solve(self, method: str = 'cg') -> np.ndarray:
        """Solve the system using specified method ('cg' or 'direct')"""
        ndof = len(self.mesh.nodes) * 3
        self.u = np.zeros(ndof)
        
        # Handle boundary conditions
        fixed_dofs = []
        prescribed_dofs = []
        prescribed_values = {}
        
        for bc in self.constraints:
            dof = 3 * bc.node_id + bc.dof
            fixed_dofs.append(dof)
        
        for bc in self.prescribed:
            dof = 3 * bc.node_id + bc.dof
            prescribed_dofs.append(dof)
            prescribed_values[dof] = bc.value
            self.u[dof] = bc.value
        
        # Get free DOFs
        all_constrained = fixed_dofs + prescribed_dofs
        free_dofs = list(set(range(ndof)) - set(all_constrained))
        
        # Create reduced system
        K_reduced = self.K[free_dofs][:, free_dofs]
        f_reduced = np.zeros(len(free_dofs))
        
        # Modified force vector computation - key change here
        for prescribed_dof, value in prescribed_values.items():
            # Use the full stiffness matrix row for proper force computation
            f_reduced -= value * self.K[free_dofs, prescribed_dof].toarray().flatten()
        
        # Print debug info
        print(f"Number of free DOFs: {len(free_dofs)}")
        print(f"Reduced system size: {K_reduced.shape}")
        print(f"First few entries of reduced force vector: {f_reduced[:5]}")
        
        # Solve using selected method
        if method == 'cg':
            from .cg_method import conjugate_gradient_solve
            x = conjugate_gradient_solve(K_reduced, f_reduced)
        elif method == 'direct':
            from .direct_method import lu_solve
            x = lu_solve(K_reduced, f_reduced)
        else:
            raise ValueError(f"Unknown solution method: {method}")
        
        # Reconstruct full solution
        for i, dof in enumerate(free_dofs):
            self.u[dof] = x[i]
        
        return self.u
    
    def _compute_shape_functions(self, xi: float, eta: float, zeta: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute shape functions and their derivatives for 8-node hex element
        Returns: (N, dN) where N are the shape functions and dN are their derivatives
        """
        # Shape functions for 8-node hex
        N = np.array([
            (1 - xi) * (1 - eta) * (1 - zeta) / 8,
            (1 + xi) * (1 - eta) * (1 - zeta) / 8,
            (1 + xi) * (1 + eta) * (1 - zeta) / 8,
            (1 - xi) * (1 + eta) * (1 - zeta) / 8,
            (1 - xi) * (1 - eta) * (1 + zeta) / 8,
            (1 + xi) * (1 - eta) * (1 + zeta) / 8,
            (1 + xi) * (1 + eta) * (1 + zeta) / 8,
            (1 - xi) * (1 + eta) * (1 + zeta) / 8
        ])
        
        # Derivatives of shape functions
        dN = np.zeros((8, 3))  # 8 nodes, 3 derivatives each (xi, eta, zeta)
        
        # dN/dxi
        dN[:, 0] = np.array([
            -(1 - eta) * (1 - zeta) / 8,
             (1 - eta) * (1 - zeta) / 8,
             (1 + eta) * (1 - zeta) / 8,
            -(1 + eta) * (1 - zeta) / 8,
            -(1 - eta) * (1 + zeta) / 8,
             (1 - eta) * (1 + zeta) / 8,
             (1 + eta) * (1 + zeta) / 8,
            -(1 + eta) * (1 + zeta) / 8
        ])
        
        # dN/deta
        dN[:, 1] = np.array([
            -(1 - xi) * (1 - zeta) / 8,
            -(1 + xi) * (1 - zeta) / 8,
             (1 + xi) * (1 - zeta) / 8,
             (1 - xi) * (1 - zeta) / 8,
            -(1 - xi) * (1 + zeta) / 8,
            -(1 + xi) * (1 + zeta) / 8,
             (1 + xi) * (1 + zeta) / 8,
             (1 - xi) * (1 + zeta) / 8
        ])
        
        # dN/dzeta
        dN[:, 2] = np.array([
            -(1 - xi) * (1 - eta) / 8,
            -(1 + xi) * (1 - eta) / 8,
            -(1 + xi) * (1 + eta) / 8,
            -(1 - xi) * (1 + eta) / 8,
             (1 - xi) * (1 - eta) / 8,
             (1 + xi) * (1 - eta) / 8,
             (1 + xi) * (1 + eta) / 8,
             (1 - xi) * (1 + eta) / 8
        ])
        
        return N, dN
    
    def _compute_jacobian(self, xi: float, eta: float, zeta: float, 
                         element_coords: np.ndarray) -> np.ndarray:
        """Compute Jacobian matrix"""
        _, dN = self._compute_shape_functions(xi, eta, zeta)
        
        # J = dN * x where x are the nodal coordinates
        J = dN.T @ element_coords
        
        return J
    
    def _compute_B_matrix(self, xi: float, eta: float, zeta: float) -> np.ndarray:
        """
        Compute strain-displacement matrix
        B matrix relates nodal displacements to strains: ε = B·u
        """
        _, dN = self._compute_shape_functions(xi, eta, zeta)
        
        # Initialize B matrix (6 strains x 24 DOFs)
        B = np.zeros((6, 24))
        
        # For each node
        for i in range(8):
            # Fill the B matrix
            # εxx = du/dx
            B[0, i*3] = dN[i, 0]
            # εyy = dv/dy
            B[1, i*3 + 1] = dN[i, 1]
            # εzz = dw/dz
            B[2, i*3 + 2] = dN[i, 2]
            # γxy = du/dy + dv/dx
            B[3, i*3] = dN[i, 1]
            B[3, i*3 + 1] = dN[i, 0]
            # γyz = dv/dz + dw/dy
            B[4, i*3 + 1] = dN[i, 2]
            B[4, i*3 + 2] = dN[i, 1]
            # γxz = du/dz + dw/dx
            B[5, i*3] = dN[i, 2]
            B[5, i*3 + 2] = dN[i, 0]
        
        return B
