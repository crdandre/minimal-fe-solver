#pragma once
#include <vector>
#include "sparse_matrix.h"

// Forward declarations
using Vector = std::vector<double>;

struct Node {
    int id;
    int component_id;
    double x, y, z;
    // Displacement vector
    double dx, dy, dz;
};

struct Element {
    int id;
    int component_id;
    // 8 nodes for hexahedral
    int nodes[8];
    // Material properties
    double E;  // Young's modulus
    double nu; // Poisson's ratio
};

struct Mesh {
    std::vector<Node> nodes;
    std::vector<Element> elements;
    // Sparse matrix for global stiffness matrix
    SparseMatrix K;
    // Global force vector
    Vector f;
};

struct BoundaryCondition {
    enum class Type {
        Fixed,
        Prescribed
    };
    
    int node_id;
    Type type;
    struct {
        double x, y, z;
    } value;
}; 