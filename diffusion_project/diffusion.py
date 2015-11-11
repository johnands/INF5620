from dolfin import *
import numpy as np

# Create mesh and define function space
#mesh = UnitSquareMesh(6, 4)
#mesh = UnitCubeMesh(6, 4, 5)

def mesh(degree, dim, Nx=5, Ny=5, Nz=5): 
    """
    Make mesh and function space V
    """
    if dim == 1:
        mesh = UnitIntervalMesh(Nx)
    elif dim == 2:
        mesh = UnitSquareMesh(Nx, Ny)
    elif dim == 3:
        mesh = UnitCubeMesh(Nx, Ny, Nz)
    else:
        raise ValueError('Dimension has to be 1, 2 or 3')

    V = FunctionSpace(mesh, 'Lagrange', degree)
    return V


# should make a solver function where I is input
def solver(I, f, alpha, V, rho=1, dt=0.5, T=2, u_0=None):
    """
    Solve diffusion equation with nonlinear constant and
    time-dependent source term
    I is initial condition, f is source term, 
    alpha is nonlinear coefficient (dependent on u)
    I and f must both be given as Expression objects
    u0 is exact solution as Expression object
    alpha is an ordinary python function
    rho is a constant
    """
    
    # load initial condition into u_1
    u_1 = interpolate(I, V)
    
    # define the unknown trial function u and test function v
    u = TrialFunction(V)
    v = TestFunction(V)
    
    # define a including 1 Picard iteration, i.e calling
    # alpha with the previous solution instead of current solution
    a = rho*u*v*dx + inner(alpha(u_1)*nabla_grad(u), nabla_grad(v))*dx
    L = (dt*f + rho*u_1)*v*dx

    u = Function(V)   # the unknown at a new time level
    t = dt

    # Initiate matrix and vector for linear system
    A = None
    b = None
    while t <= T:
        A = assemble(a, tensor=A)
        b = assemble(L, tensor=b)
        solve(A, u.vector(), b)

        t += dt
        u_1.assign(u)

        if u_e is not None and t == 3*dt:
            # project exact solution onto V
            u0.t = t
            u_e = interpolate(u0, V)
            e = u_e.vector().array() - u.vector().array()
            E = np.sqrt(np.sum(e**2)/u.vector().array().size)
            print E/dt
            
    u = u_1
    return u.vector().array(), E


def test_constant(C):
    """Verification using a constant solution"""

    # Initiate I and f as Constant objects
    I = Constant(C)
    f = Constant(0.0)  

    # alpha is an ordinary python function
    alpha = lambda u: u**2 + u
    rho = 5.0

    # numerical parameters
    dt = 0.5
    dx = 0.5; dy = 0.5
    Nx = int(round(1./dx))
    Ny = int(round(1./dy))
    T = 2.0

    # make function space
    V = mesh(1, 2, Nx, Ny)

    # solve
    u = solver(I, f, alpha, V, rho, dt, T)
    
    # verify
    tol = 1E-12
    diff = abs(u-C)
    assert diff.max() < tol


def error_measure():
    """Verification using error measure"""

    u0 = Expression('exp(-pi**2*t)*cos(pi*x[0])')
    I = Expression('cos(pi*x[0])')
    f = Constant(0.0)

    alpha = lambda u: 1.0
    rho = 1.0

    # numerical parameters
    h = 0.5			# common discretization parameter
    Kt = 1; Kx = 1; Ky = 1
    K = Kt + Kx + Ky
    Nx = int(round(1./np.sqrt(h)))
    Ny = int(round(1./np.sqrt(h)))
    T = 2.0
    dt = h

    # make function space
    V = mesh(1, 2, Nx, Ny)

    # solve
    u = solver(I, f, alpha, V, rho, dt, T, u_0)


if __name__ == '__main__':
    #test_constant(4.0)
    error_measure()

    
    








