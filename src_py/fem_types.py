from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import numpy as np

@dataclass
class Node:
    id: int
    component_id: int
    x: float
    y: float
    z: float

@dataclass
class Element:
    id: int
    component_id: int
    nodes: List[int]  # List of 8 node IDs for hex element

@dataclass
class MaterialProperties:
    E: float  # Young's modulus
    nu: float  # Poisson's ratio
    D: Optional[np.ndarray] = None  # Constitutive matrix (computed internally)

@dataclass
class BoundaryCondition:
    node_id: int
    dof: int  # 0, 1, or 2 for x, y, z
    value: float  # 0.0 for fixed constraint, otherwise prescribed displacement

class MeshData:
    def __init__(self, nodes: List[Node], elements: List[Element]):
        self.nodes = nodes
        self.elements = elements
        self.node_coords: np.ndarray  # Nx3 array of coordinates
        self.element_connectivity: np.ndarray  # Mx8 array of node indices
        self._process_mesh()
    
    def _process_mesh(self):
        # Convert nodes to coordinate array
        self.node_coords = np.array([[n.x, n.y, n.z] for n in self.nodes])
        # Create element connectivity array
        self.element_connectivity = np.array([e.nodes for e in self.elements])
