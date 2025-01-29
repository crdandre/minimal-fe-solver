#include "element.h"
#include <array>
#include <cmath>

// Define the global variables
const std::array<double, 2> gauss_points = {-1.0/std::sqrt(3.0), 1.0/std::sqrt(3.0)};
const double gauss_weight = 1.0;

void calculateShapeFunctions(double xi, double eta, double zeta, 
                           double N[8], double dN[8][3]) {
    // Shape functions
    N[0] = 0.125 * (1 - xi) * (1 - eta) * (1 - zeta);
    N[1] = 0.125 * (1 + xi) * (1 - eta) * (1 - zeta);
    N[2] = 0.125 * (1 + xi) * (1 + eta) * (1 - zeta);
    N[3] = 0.125 * (1 - xi) * (1 + eta) * (1 - zeta);
    N[4] = 0.125 * (1 - xi) * (1 - eta) * (1 + zeta);
    N[5] = 0.125 * (1 + xi) * (1 - eta) * (1 + zeta);
    N[6] = 0.125 * (1 + xi) * (1 + eta) * (1 + zeta);
    N[7] = 0.125 * (1 - xi) * (1 + eta) * (1 + zeta);
    
    // Shape function derivatives
    // dN/dxi
    dN[0][0] = -0.125 * (1 - eta) * (1 - zeta);
    dN[1][0] =  0.125 * (1 - eta) * (1 - zeta);
    dN[2][0] =  0.125 * (1 + eta) * (1 - zeta);
    dN[3][0] = -0.125 * (1 + eta) * (1 - zeta);
    dN[4][0] = -0.125 * (1 - eta) * (1 + zeta);
    dN[5][0] =  0.125 * (1 - eta) * (1 + zeta);
    dN[6][0] =  0.125 * (1 + eta) * (1 + zeta);
    dN[7][0] = -0.125 * (1 + eta) * (1 + zeta);

    // dN/deta
    dN[0][1] = -0.125 * (1 - xi) * (1 - zeta);
    dN[1][1] = -0.125 * (1 + xi) * (1 - zeta);
    dN[2][1] =  0.125 * (1 + xi) * (1 - zeta);
    dN[3][1] =  0.125 * (1 - xi) * (1 - zeta);
    dN[4][1] = -0.125 * (1 - xi) * (1 + zeta);
    dN[5][1] = -0.125 * (1 + xi) * (1 + zeta);
    dN[6][1] =  0.125 * (1 + xi) * (1 + zeta);
    dN[7][1] =  0.125 * (1 - xi) * (1 + zeta);

    // dN/dzeta
    dN[0][2] = -0.125 * (1 - xi) * (1 - eta);
    dN[1][2] = -0.125 * (1 + xi) * (1 - eta);
    dN[2][2] = -0.125 * (1 + xi) * (1 + eta);
    dN[3][2] = -0.125 * (1 - xi) * (1 + eta);
    dN[4][2] =  0.125 * (1 - xi) * (1 - eta);
    dN[5][2] =  0.125 * (1 + xi) * (1 - eta);
    dN[6][2] =  0.125 * (1 + xi) * (1 + eta);
    dN[7][2] =  0.125 * (1 - xi) * (1 + eta);
} 