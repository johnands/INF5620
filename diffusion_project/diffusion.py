from dolfin import *
import numpy as np
import sys

# Create mesh and define function space
#mesh = UnitSquareMesh(6, 4)
#mesh = UnitCubeMesh(6, 4, 5)

# parametrizing the number of space dimensions
degree = int(sys.argv[1])
divisions = [int(arg) for arg in sys.argv[2:]]
d = len(divisions)
domain_type = [UnitIntervalMesh, UnitSquareMesh, UnitCubeMesh]
mesh = domain_type[d-1](*divisions)
V = FunctionSpace(mesh, 'Lagrange', degree)
