#include "types.h"
#include "solver.h"
#include <iostream>
#include <iomanip>  // for std::setprecision
#include <fstream>
#include <filesystem>

// Helper function to create a 2x2x2 cube mesh
Mesh createCubeMesh() {
    // Initialize mesh with proper matrix size (81x81 for 27 nodes * 3 DOF)
    Mesh mesh{std::vector<Node>(), std::vector<Element>(), SparseMatrix(81, 81), std::vector<double>(81, 0.0)};
    
    // Create 27 nodes for a 2x2x2 cube (3x3x3 grid)
    std::vector<Node> nodes;
    int node_id = 0;
    for(int z = 0; z < 3; z++) {
        for(int y = 0; y < 3; y++) {
            for(int x = 0; x < 3; x++) {
                nodes.push_back({
                    node_id,    // id
                    0,          // component_id
                    x * 0.5,    // x (scaled to [0, 1])
                    y * 0.5,    // y (scaled to [0, 1])
                    z * 0.5,    // z (scaled to [0, 1])
                    0.0, 0.0, 0.0  // dx, dy, dz
                });
                node_id++;
            }
        }
    }
    mesh.nodes = nodes;
    
    // Create 8 hexahedral elements
    for(int ez = 0; ez < 2; ez++) {
        for(int ey = 0; ey < 2; ey++) {
            for(int ex = 0; ex < 2; ex++) {
                Element element;
                element.id = ex + ey * 2 + ez * 4;
                element.component_id = 0;
                element.E = 200e9;  // Steel-like material
                element.nu = 0.3;   // Typical Poisson's ratio
                
                // Calculate node indices for this element
                int base = ex + ey * 3 + ez * 9;
                element.nodes[0] = base;
                element.nodes[1] = base + 1;
                element.nodes[2] = base + 4;
                element.nodes[3] = base + 3;
                element.nodes[4] = base + 9;
                element.nodes[5] = base + 10;
                element.nodes[6] = base + 13;
                element.nodes[7] = base + 12;
                
                mesh.elements.push_back(element);
            }
        }
    }
    
    return mesh;
}

// Add this helper function before main()
void writeVTKFile(const Mesh& mesh, int timestep) {
    // Create output directory if it doesn't exist
    std::string output_dir = "output";
    std::filesystem::create_directory(output_dir);
    
    // Construct full path
    std::string filename = output_dir + "/cube_" + std::to_string(timestep) + ".vtk";
    std::ofstream file(filename);
    
    // VTK header
    file << "# vtk DataFile Version 2.0\n";
    file << "Cube deformation\n";
    file << "ASCII\n";
    file << "DATASET UNSTRUCTURED_GRID\n";
    
    // Write points (nodes with displacements)
    file << "POINTS " << mesh.nodes.size() << " float\n";
    for (const auto& node : mesh.nodes) {
        file << node.x + node.dx << " " 
             << node.y + node.dy << " "
             << node.z + node.dz << "\n";
    }
    
    // Write cell (element) connectivity
    file << "\nCELLS " << mesh.elements.size() << " " << mesh.elements.size() * 9 << "\n";
    for (const auto& element : mesh.elements) {
        file << "8";
        for (int i = 0; i < 8; i++) {
            file << " " << element.nodes[i];
        }
        file << "\n";
    }
    
    // Write cell types
    file << "\nCELL_TYPES " << mesh.elements.size() << "\n";
    for (size_t i = 0; i < mesh.elements.size(); i++) {
        file << "12\n";  // 12 = VTK_HEXAHEDRON
    }
    
    file.close();
}

int main() {
    // Create test mesh
    Mesh mesh = createCubeMesh();
    
    // Create solver
    FESolver solver(mesh);
    
    // Define timesteps and max displacement
    const int num_steps = 20;  // increased from 5 to 20
    const double max_displacement = 0.1;  // meters
    
    for(int step = 0; step < num_steps; step++) {
        // Calculate current displacement
        double current_disp = (step + 1) * max_displacement / num_steps;
        
        std::cout << "\n=== Timestep " << std::setw(2) << step + 1 << "/" << num_steps 
                  << " (z-displacement = " << std::fixed << std::setprecision(4) << current_disp << "m) ===" << std::endl;
        
        // Define boundary conditions
        std::vector<BoundaryCondition> bcs;
        
        // Fix bottom nodes (z = 0)
        for(int i = 0; i < 9; i++) {  // 9 nodes on bottom face (3x3)
            bcs.push_back({i, BoundaryCondition::Type::Fixed, {0.0, 0.0, 0.0}});
        }
        
        // Apply displacement to top nodes (z = 1)
        for(int i = 18; i < 27; i++) {  // 9 nodes on top face (3x3)
            bcs.push_back({i, BoundaryCondition::Type::Prescribed, {0.0, 0.0, current_disp}});
        }
        
        // Solve
        solver.solve(bcs);
        
        // Print results with better formatting
        std::cout << "\nResults Summary:" << std::endl;
        std::cout << std::fixed << std::setprecision(6);
        
        std::cout << "Bottom nodes (fixed):" << std::endl;
        for(int i = 0; i < 9; i++) {
            const auto& node = mesh.nodes[i];
            std::cout << "  Node " << i << ": (" 
                     << node.dx << ", " 
                     << node.dy << ", " 
                     << node.dz << ")" << std::endl;
        }
        
        std::cout << "\nTop nodes (prescribed z-displacement):" << std::endl;
        for(int i = 18; i < 27; i++) {
            const auto& node = mesh.nodes[i];
            std::cout << "  Node " << i << ": (" 
                     << node.dx << ", " 
                     << node.dy << ", " 
                     << node.dz << ")" << std::endl;
        }
        
        // After solving and printing results, write VTK file
        writeVTKFile(mesh, step);
    }
    
    return 0;
} 