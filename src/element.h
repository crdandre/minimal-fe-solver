#pragma once
#include <array>

extern const std::array<double, 2> gauss_points;
extern const double gauss_weight;

void calculateShapeFunctions(double xi, double eta, double zeta, 
                           double N[8], double dN[8][3]); 