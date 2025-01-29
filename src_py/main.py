from typing import List, Tuple
import numpy as np
import os
import argparse
from .fem_types import Node, Element, MeshData
from src_py.solver import FEMSolver

def create_cube_mesh() -> MeshData:
    """Create a 2x2x2 cube mesh (3x3x3 nodes, 8 elements)"""
    # Create 27 nodes (3x3x3 grid)
    nodes = []
    node_id = 0
    for z in range(3):
        for y in range(3):
            for x in range(3):
                nodes.append(Node(
                    id=node_id,
                    component_id=0,
                    x=x * 0.5,  # Scale to [0, 1]
                    y=y * 0.5,
                    z=z * 0.5
                ))
                node_id += 1

    # Create 8 hexahedral elements (2x2x2)
    elements = []
    for ez in range(2):
        for ey in range(2):
            for ex in range(2):
                # Calculate element ID
                elem_id = ex + ey * 2 + ez * 4
                
                # Calculate base node index
                base = ex + ey * 3 + ez * 9
                
                # Create element connectivity (8 nodes)
                element_nodes = [
                    base,      # n0
                    base + 1,  # n1
                    base + 4,  # n2
                    base + 3,  # n3
                    base + 9,  # n4
                    base + 10, # n5
                    base + 13, # n6
                    base + 12  # n7
                ]
                
                elements.append(Element(
                    id=elem_id,
                    component_id=0,
                    nodes=element_nodes
                ))

    return MeshData(nodes, elements)

def solve_fem(mesh: MeshData, 
             materials: List[Tuple[int, float, float]], 
             constraints: List[Tuple[int, int]], 
             prescribed: List[Tuple[int, int, float]],
             solution_method: str = 'direct',
             num_steps: int = 10) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Solve FEM problem with intermediate steps
    
    Args:
        mesh: MeshData object
        materials: List of (component_id, E, nu)
        constraints: List of (node_id, dof)
        prescribed: List of (node_id, dof, value)
        num_steps: Number of intermediate steps
        
    Returns:
        List of (initial_positions, intermediate_positions) for each step
    """
    solver = FEMSolver(mesh)
    initial_positions = np.array([[node.x, node.y, node.z] for node in mesh.nodes])
    
    for comp_id, E, nu in materials:
        solver.add_material(comp_id, E, nu)
    
    for node_id, dof in constraints:
        solver.add_constraint(node_id, dof)
    
    results = []
    # Solve in steps
    for step in range(num_steps + 1):
        # Create a new solver for each step to avoid prescribed displacement accumulation
        solver = FEMSolver(mesh)
        
        # Re-add materials and constraints
        for comp_id, E, nu in materials:
            solver.add_material(comp_id, E, nu)
        
        for node_id, dof in constraints:
            solver.add_constraint(node_id, dof)
        
        # Scale prescribed displacements by step
        scale = step / num_steps
        for node_id, dof, value in prescribed:
            solver.add_prescribed_displacement(node_id, dof, value * scale)
        
        solver.assemble_system()
        displacements = solver.solve(method=solution_method)
        final_positions = initial_positions + displacements.reshape(-1, 3)
        results.append((initial_positions, final_positions))
    
    return results

def main():
    # Create test mesh
    mesh = create_cube_mesh()
    
    # Define materials (Steel-like)
    materials = [(0, 200e9, 0.3)]  # component_id, E, nu
    
    # Define constraints
    constraints = []
    prescribed = []
    
    # Fix bottom nodes (z = 0)
    for i in range(9):  # 9 nodes on bottom face (3x3)
        constraints.extend([
            (i, 0),  # Fix x
            (i, 1),  # Fix y
            (i, 2)   # Fix z
        ])
    
    # Apply displacement to top nodes (z = 1) with small x and y components
    displacement = 0.1  # primary strain
    off_axis = 0.1  # small off-axis component
    for i in range(18, 27):  # 9 nodes on top face (3x3)
        prescribed.extend([
            (i, 0, off_axis),        # Small x displacement
            (i, 1, off_axis),        # Small y displacement
            (i, 2, displacement)      # Main z displacement
        ])
    
    # Solve using both methods
    print("\nDirect Method Results:")
    results_direct = solve_fem(mesh, materials, constraints, prescribed, 'direct')
    for step, (initial_positions, final_positions) in enumerate(results_direct):
        print(f"\nStep {step}:")
        print_results(mesh, final_positions)
    
    print("\nConjugate Gradient Method Results:")
    results_cg = solve_fem(mesh, materials, constraints, prescribed, 'cg')
    for step, (initial_positions, final_positions) in enumerate(results_cg):
        print(f"\nStep {step}:")
        print_results(mesh, final_positions)
    
    # Compare results
    diff = np.max(np.abs(results_direct[-1][1] - results_cg[-1][1]))
    print(f"\nMaximum difference between methods: {diff:.2e}")

def print_results(mesh, displacements):
    print("\nDisplacements at each node:")
    for i in range(len(mesh.nodes)):
        dx = displacements[i, 0]  # Access first column (x displacement)
        dy = displacements[i, 1]  # Access second column (y displacement)
        dz = displacements[i, 2]  # Access third column (z displacement)
        print(f"Node {i}: ({dx:.6f}, {dy:.6f}, {dz:.6f})")

if __name__ == "__main__":
    main()
