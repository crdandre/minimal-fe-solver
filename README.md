# Minimal FE Solvers

This is mostly claude-generated for now (Jan 25, 2025), but am trying to get the hang of a minimal FE solver that can be compiled into WASM and run as a demo on my personal website. There are probably implementations for this elsewhere but trying to get the hang of the basics from scratch.

## Architecture for now
- 3D mesh
- 8-node hexahedra
- linear elastic materials
- single component
- fixed node constraints
- prescribed displacement
- conjugate gradient solver

## Build and run
```
mkdir build
cd build
cmake ..
make
./fesolver
```

## Notes
- Hourglass mode handling not added, occurs in hex meshes with forces directly along the element edges
- Terms: condition number (related to stability)