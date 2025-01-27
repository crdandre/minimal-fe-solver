#pragma once
#include "types.h"
#include <vector>

using Matrix = std::vector<std::vector<double>>;

class FESolver {
public:
    FESolver(Mesh& mesh);
    
    // Solve system using Conjugate Gradient
    void solve(const std::vector<BoundaryCondition>& bcs);
    
private:
    // Assemble global stiffness matrix
    void assembleStiffnessMatrix();
    
    // Calculate element stiffness matrix
    Matrix calculateElementStiffness(const Element& elem);
    
    // Conjugate Gradient solver
    std::vector<double> conjugateGradient(const SparseMatrix& A, const std::vector<double>& b, double tol = 1e-6);
    
    Mesh& mesh_;
    SparseMatrix K_;
    std::vector<double> f_;
}; 