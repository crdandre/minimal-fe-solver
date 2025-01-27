#include "solver.h"
#include "element.h"
#include <cmath>
#include <iostream> // For debugging
#include <numeric>  // for std::inner_product
#include <algorithm>  // for std::find

FESolver::FESolver(Mesh& mesh) 
    : mesh_(mesh),
      K_(3 * mesh.nodes.size(), 3 * mesh.nodes.size()),  // Initialize K_ with proper dimensions
      f_(3 * mesh.nodes.size(), 0.0)  // Initialize f_ with zeros
{
    assembleStiffnessMatrix();
}

void FESolver::assembleStiffnessMatrix() {
    for (const auto& elem : mesh_.elements) {
        auto Ke = calculateElementStiffness(elem);
        
        // Assembly into global matrix
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                for (int di = 0; di < 3; di++) {
                    for (int dj = 0; dj < 3; dj++) {
                        int gi = 3 * elem.nodes[i] + di;
                        int gj = 3 * elem.nodes[j] + dj;
                        K_.add(gi, gj, Ke[3*i + di][3*j + dj]);
                    }
                }
            }
        }
    }
}

Matrix FESolver::calculateElementStiffness(const Element& elem) {
    Matrix Ke(24, std::vector<double>(24, 0.0));
    
    // Material matrix D for plane strain
    Matrix D(6, std::vector<double>(6, 0.0));
    double E = elem.E;
    double nu = elem.nu;
    double factor = E / ((1 + nu) * (1 - 2*nu));
    
    D[0][0] = D[1][1] = D[2][2] = factor * (1 - nu);
    D[0][1] = D[0][2] = D[1][0] = D[1][2] = D[2][0] = D[2][1] = factor * nu;
    D[3][3] = D[4][4] = D[5][5] = factor * (1 - 2*nu) / 2;

    std::cout << "Element E: " << E << ", nu: " << nu << std::endl;
    
    // Gauss quadrature
    for (double xi : gauss_points) {
        for (double eta : gauss_points) {
            for (double zeta : gauss_points) {
                double N[8];
                double dN[8][3];
                calculateShapeFunctions(xi, eta, zeta, N, dN);
                
                // Calculate Jacobian
                std::vector<std::vector<double>> J(3, std::vector<double>(3, 0.0));
                for (int i = 0; i < 8; i++) {
                    const auto& node = mesh_.nodes[elem.nodes[i]];
                    J[0][0] += dN[i][0] * node.x;  J[0][1] += dN[i][0] * node.y;  J[0][2] += dN[i][0] * node.z;
                    J[1][0] += dN[i][1] * node.x;  J[1][1] += dN[i][1] * node.y;  J[1][2] += dN[i][1] * node.z;
                    J[2][0] += dN[i][2] * node.x;  J[2][1] += dN[i][2] * node.y;  J[2][2] += dN[i][2] * node.z;
                }
                
                // Calculate determinant
                double detJ = J[0][0]*(J[1][1]*J[2][2] - J[1][2]*J[2][1])
                           - J[0][1]*(J[1][0]*J[2][2] - J[1][2]*J[2][0])
                           + J[0][2]*(J[1][0]*J[2][1] - J[1][1]*J[2][0]);
                
                std::cout << "detJ: " << detJ << std::endl;
                
                // Calculate inverse of Jacobian
                std::vector<std::vector<double>> Jinv(3, std::vector<double>(3, 0.0));
                Jinv[0][0] = (J[1][1]*J[2][2] - J[1][2]*J[2][1])/detJ;
                Jinv[0][1] = (J[0][2]*J[2][1] - J[0][1]*J[2][2])/detJ;
                Jinv[0][2] = (J[0][1]*J[1][2] - J[0][2]*J[1][1])/detJ;
                Jinv[1][0] = (J[1][2]*J[2][0] - J[1][0]*J[2][2])/detJ;
                Jinv[1][1] = (J[0][0]*J[2][2] - J[0][2]*J[2][0])/detJ;
                Jinv[1][2] = (J[0][2]*J[1][0] - J[0][0]*J[1][2])/detJ;
                Jinv[2][0] = (J[1][0]*J[2][1] - J[1][1]*J[2][0])/detJ;
                Jinv[2][1] = (J[0][1]*J[2][0] - J[0][0]*J[2][1])/detJ;
                Jinv[2][2] = (J[0][0]*J[1][1] - J[0][1]*J[1][0])/detJ;
                
                // Calculate B matrix
                std::vector<std::vector<double>> B(6, std::vector<double>(24, 0.0));
                for (int i = 0; i < 8; i++) {
                    // Calculate derivatives in global coordinates
                    double dNx = Jinv[0][0]*dN[i][0] + Jinv[0][1]*dN[i][1] + Jinv[0][2]*dN[i][2];
                    double dNy = Jinv[1][0]*dN[i][0] + Jinv[1][1]*dN[i][1] + Jinv[1][2]*dN[i][2];
                    double dNz = Jinv[2][0]*dN[i][0] + Jinv[2][1]*dN[i][1] + Jinv[2][2]*dN[i][2];
                    
                    // Fill B matrix
                    int col = 3*i;
                    B[0][col] = dNx;
                    B[1][col+1] = dNy;
                    B[2][col+2] = dNz;
                    B[3][col] = dNy; B[3][col+1] = dNx;
                    B[4][col+1] = dNz; B[4][col+2] = dNy;
                    B[5][col] = dNz; B[5][col+2] = dNx;
                }
                
                // Calculate B^T * D * B * detJ * weight
                double weight = gauss_weight * gauss_weight * gauss_weight;
                for (int i = 0; i < 24; i++) {
                    for (int j = 0; j < 24; j++) {
                        double sum = 0.0;
                        for (int k = 0; k < 6; k++) {
                            for (int l = 0; l < 6; l++) {
                                sum += B[k][i] * D[k][l] * B[l][j];
                            }
                        }
                        Ke[i][j] += sum * detJ * weight;
                    }
                }
            }
        }
    }
    
    // Print first few entries of Ke for debugging
    std::cout << "Ke[0][0]: " << Ke[0][0] << std::endl;
    std::cout << "Ke[1][1]: " << Ke[1][1] << std::endl;
    std::cout << "Ke[2][2]: " << Ke[2][2] << std::endl;
    
    return Ke;
}

void FESolver::solve(const std::vector<BoundaryCondition>& bcs) {
    int ndof = 3 * mesh_.nodes.size();
    std::vector<double> u(ndof, 0.0);
    std::vector<bool> fixed(ndof, false);
    std::vector<bool> prescribed(ndof, false);
    
    // Store boundary conditions
    for (const auto& bc : bcs) {
        int idx = 3 * bc.node_id;
        if (bc.type == BoundaryCondition::Type::Fixed) {
            fixed[idx] = fixed[idx+1] = fixed[idx+2] = true;
            std::cout << "Fixed DOFs: " << idx << ", " << idx+1 << ", " << idx+2 << std::endl;
        } else {
            // Only mark z-component as prescribed for top nodes
            u[idx+2] = bc.value.z;  // Only set z-displacement
            prescribed[idx+2] = true;  // Only mark z-component as prescribed
            std::cout << "Prescribed displacement at DOF: " << idx+2 
                     << " = " << bc.value.z << " (z-direction)" << std::endl;
        }
    }
    
    // Create reduced system for free DOFs
    std::vector<int> free_dofs;
    for (int i = 0; i < ndof; i++) {
        if (!fixed[i] && !prescribed[i]) {
            free_dofs.push_back(i);
        }
    }
    
    int n_free = free_dofs.size();
    std::cout << "Number of free DOFs: " << n_free << std::endl;
    
    // Create reduced stiffness matrix and force vector
    SparseMatrix K_red(n_free, n_free);
    std::vector<double> f_red(n_free, 0.0);
    
    // Build reduced stiffness matrix
    for (size_t i = 0; i < n_free; i++) {
        int row = free_dofs[i];
        for (size_t j = 0; j < n_free; j++) {
            int col = free_dofs[j];
            // Copy corresponding entry from full matrix
            for (int k = K_.row_ptr[row]; k < K_.row_ptr[row+1]; k++) {
                if (K_.col_indices[k] == col) {
                    K_red.add(i, j, K_.values[k]);
                    break;
                }
            }
        }
    }
    
    // Compute reduced force vector: f_red = f - K * u_prescribed
    for (size_t i = 0; i < n_free; i++) {
        int row = free_dofs[i];
        f_red[i] = f_[row];
        
        for (int j = K_.row_ptr[row]; j < K_.row_ptr[row+1]; j++) {
            int col = K_.col_indices[j];
            if (prescribed[col]) {
                f_red[i] -= K_.values[j] * u[col];
            }
        }
    }
    
    // Print reduced system size and some values
    std::cout << "Reduced system size: " << n_free << std::endl;
    std::cout << "First few entries of reduced force vector:" << std::endl;
    for (size_t i = 0; i < std::min(size_t(5), f_red.size()); i++) {
        std::cout << "f_red[" << i << "] = " << f_red[i] << std::endl;
    }
    
    // Solve reduced system
    std::vector<double> result;
    if (n_free > 0) {
        result = conjugateGradient(K_red, f_red, 1e-6);
    }
    
    // Reconstruct full solution with more detailed debugging
    std::vector<double> u_full(ndof, 0.0);
    
    std::cout << "\nSolution reconstruction:" << std::endl;
    
    // First set prescribed values
    for (int i = 0; i < ndof; i++) {
        if (prescribed[i]) {
            u_full[i] = u[i];
            std::cout << "DOF " << i << " (prescribed): " << u_full[i] << std::endl;
        }
    }
    
    // Then set free DOF values
    for (size_t i = 0; i < free_dofs.size(); i++) {
        int dof = free_dofs[i];
        u_full[dof] = result[i];
        std::cout << "DOF " << dof << " (free): " << u_full[dof] << std::endl;
    }
    
    // Fixed DOFs remain 0.0
    for (int i = 0; i < ndof; i++) {
        if (fixed[i]) {
            std::cout << "DOF " << i << " (fixed): " << u_full[i] << std::endl;
        }
    }
    
    // Update nodal displacements in both the solver's mesh and the input mesh
    for (size_t i = 0; i < mesh_.nodes.size(); i++) {
        int idx = 3 * i;
        mesh_.nodes[i].dx = u_full[idx];
        mesh_.nodes[i].dy = u_full[idx+1];
        mesh_.nodes[i].dz = u_full[idx+2];
        
        // Also update the input mesh through the reference
        mesh_.nodes[i] = mesh_.nodes[i];  // This copies back to the original mesh
    }
    
    std::cout << "\nFinal nodal displacements (solver):" << std::endl;
    for (const auto& node : mesh_.nodes) {
        std::cout << "Node " << node.id << ": ("
                 << node.dx << ", " 
                 << node.dy << ", " 
                 << node.dz << ")" << std::endl;
    }
}

std::vector<double> FESolver::conjugateGradient(const SparseMatrix& A, const std::vector<double>& b, double tol) {
    int n = b.size();
    std::vector<double> x(n, 0.0);  // Initial guess
    std::vector<double> r = b;      // Initial residual
    
    // Debug prints
    std::cout << "\nConjugate Gradient Debug:" << std::endl;
    std::cout << "System size: " << n << std::endl;
    std::cout << "Initial force vector (b) norm: " << std::sqrt(std::inner_product(b.begin(), b.end(), b.begin(), 0.0)) << std::endl;
    
    // Check if force vector is all zeros
    bool all_zeros = true;
    for (double val : b) {
        if (std::abs(val) > 1e-10) {
            all_zeros = false;
            break;
        }
    }
    if (all_zeros) {
        std::cout << "WARNING: Force vector is all zeros!" << std::endl;
        return x;
    }
    
    std::vector<double> p = r;      // Initial search direction
    
    double rsold = std::inner_product(r.begin(), r.end(), r.begin(), 0.0);
    std::cout << "Initial residual norm: " << std::sqrt(rsold) << std::endl;
    
    for (int iter = 0; iter < n && iter < 1000; iter++) {
        std::vector<double> Ap = A * p;
        double alpha = rsold / std::inner_product(p.begin(), p.end(), Ap.begin(), 0.0);
        
        for (int i = 0; i < n; i++) {
            x[i] += alpha * p[i];
            r[i] -= alpha * Ap[i];
        }
        
        double rsnew = std::inner_product(r.begin(), r.end(), r.begin(), 0.0);
        std::cout << "Iteration " << iter << ", residual: " << std::sqrt(rsnew) << std::endl;
        
        if (std::sqrt(rsnew) < tol) {
            std::cout << "Converged after " << iter + 1 << " iterations" << std::endl;
            break;
        }
        
        double beta = rsnew / rsold;
        for (int i = 0; i < n; i++) {
            p[i] = r[i] + beta * p[i];
        }
        
        rsold = rsnew;
    }
    
    // Print final solution norm
    std::cout << "Solution norm: " << std::sqrt(std::inner_product(x.begin(), x.end(), x.begin(), 0.0)) << std::endl;
    
    return x;
} 