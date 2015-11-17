"""
###############################################################

Solving nonlinear diffusion equation using FEniCS.
Uncomment function calls at bottom of script to 
run different kinds of verifications and visualizations.

###############################################################
"""




from dolfin import *
import numpy as np


def function_space(degree, dim, Nx=5, Ny=5, Nz=5): 
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
    return V, mesh


# should make a solver function where I is input
def solver(f, alpha, V, rho, dt, T, u0, animate=False):
    """
    Solve diffusion equation with nonlinear diffusion 
    coefficient and time-dependent source term
    - u0: exact solution, u0(t=0) = I
    - f: source term
    - alpha: nonlinear diffusion coefficient
    - rho: constant
    - V: function space which u is defined on
    - dt: time step, T: stopping time
    """
    
    # load initial condition into u_1
    u_1 = interpolate(u0, V)

    # plot   
    if animate:
        i = 0
        fig = plot(u_1)
        fig.set_min_max(0, 0.5*u_1.vector().array().max())
        fig.plot(u_1)
        #interactive()
        #fig.write_png('diff_%d04d' % i)
    
    # define the unknown trial function u and test function v
    u = TrialFunction(V)
    v = TestFunction(V)
    
    # define a including 1 Picard iteration, i.e calling
    # alpha with the previous solution instead of current solution
    a = u*v*dx + inner(dt*alpha(u_1)*nabla_grad(u)/rho, nabla_grad(v))*dx
    L = (dt/rho)*f*v*dx + u_1*v*dx

    u = Function(V)   # the unknown at a new time level
    t = dt

    # time loop
    while t <= T:
        u0.t = t
        f.t = t
        A = assemble(a)
        b = assemble(L)
        solve(A, u.vector(), b)

        #u_e = interpolate(u0, V)
        
        if animate:
            i += 1
            fig.plot(u)
            #interactive()
            #fig.write_png('diff_%d04d' % i)
        
        u_1.assign(u)
        t += dt   
    u = u_1
    return u


def test_constant():
    """Verification: constant solution, u_e = C"""

    # Exact solution
    C = 2.0
    u0 = Expression('C', C=C)

    # source term
    f = Constant(0.0)  

    # diffusion coefficient
    alpha = lambda u: u**2 + u

    rho = 5.0

    # numerical parameters
    dt = 0.5
    dx = 0.5; dy = 0.5; dz = 0.5
    Nx = int(round(1./dx))
    Ny = int(round(1./dy))
    Nz = int(round(1./dz))
    T = 2.0

    Lagrange_degree = [1, 2]	# degree of Lagrange basis functions
    Dimension = [1, 2, 3]	# 1d, 2d, 3d
    
    for degree in Lagrange_degree:
        for dim in Dimension:

            # make function space
            V, mesh = function_space(degree, dim, Nx, Ny, Nz)

            # solve
            u = solver(f, alpha, V, rho, dt, T, u0)
    
            # verify
            tol = 1E-12
            diff = abs(u.vector().array() - C)
            assert diff.max() < tol


def error_measure():
    """Verification using error measure"""

    # exact solution
    u0 = Expression('exp(-pow(pi,2)*t)*cos(pi*x[0])', t=0)

    # source term
    f = Constant(0.0)
    
    # time point when E is measured
    T = 0.5    

    # diffusion coefficient
    alpha = lambda x: 1.0

    rho = Constant(1.0)
 
    # compute L2 norm of E, mesh is doubly refined for each iteration
    h_values = [0.5*2**(-i) for i in range(8)]
    for h in h_values:
        # mesh parameters
        Nx = int(round(1./np.sqrt(h)))
        Ny = int(round(1./np.sqrt(h)))
        dt = h

        # make function space
        V, mesh = function_space(1, 2, Nx, Ny)

        u0.t = 0	# initial condition

        # solve
        u = solver(f, alpha, V, rho, dt, T, u0)

        # compute L2 norm     
        u0.t = T
        u_e = interpolate(u0, V)
        e = u_e.vector().array() - u.vector().array()
        E = np.sqrt(np.sum(e**2)/u.vector().array().size)
        print 'E = %.6f, h = %.6f: K = %.6f' % (E, h, E/h)


def manufactured_solution():
    """
    Compare u and u_exact for a manufactured solution
    """

    # nonlinear diffusion coefficient
    alpha = lambda u: 1 + u**2

    rho = Constant(1.0)

    # source term (computed with sympy)
    f = Expression('-rho*pow(x[0],3)/3. + rho*pow(x[0],2)/2. + 8*pow(t,3)*pow(x[0],7)/9. -' + 
                   '28*pow(t,3)*pow(x[0],6)/9. + 7*pow(t,3)*pow(x[0],5)/2. - 5*pow(t,3)*pow(x[0],4)/4.' +
                   ' + 2*t*x[0] - t', rho=rho, t=0)

    # manufactured solution
    u0 = Expression('t*pow(x[0],2)*(0.5 - x[0]/3)', t=0)

    # numerical paramters
    h = 0.05
    dt = h
    dx = sqrt(h)
    Nx = int(round(1./dx))
    Tlist = [0.1, 0.5, 1.0]

    # compare u and u_exact for different t
    for T in Tlist:
        # set t to zero before solving
        u0.t = 0
        f.t = 0

        # make function space and solve
        V, mesh = function_space(1, 1, Nx) 
        u = solver(f, alpha, V, rho, dt, T, u0)

        # project exact solution onto V
        u0.t = T
        u_e = interpolate(u0, V)

        e = u_e.vector().array() - u.vector().array()
        E = np.sqrt(np.sum(e**2)/u.vector().array().size)
        print 'E = %.9f, h = %.6f: t = %.6f' % (E, h, T)


    # plot u and u_exact
    #plot(u, u_e, title='t')
    #plot(u_e, title='u_e')
    #interactive()


def convergence_rate():

    # nonlinear diffusion coefficient
    alpha = lambda u: 1 + u**2

    rho = Constant(1.0)

    # source term
    f = Expression('rho*pow(x[0],2)*(-2*x[0] + 3)/6. - ' + 
                   '(-12*t*x[0] + 3*t*(-2*x[0] + 3))*(pow(x[0],4)*pow(-dt+t,2)*pow(-2*x[0]+3,2) + 36)/324.' +
                   ' - (-6*t*pow(x[0],2) + 6*t*x[0]*(-2*x[0] + 3))*(36*pow(x[0],4)*pow(-dt+t,2)*(2*x[0] - 3)' +
                   ' + 36*pow(x[0],3)*pow(-dt+t,2)*pow(-2*x[0]+3,2))/5832.', rho=rho, t=0, dt=0.5)

    # manufactured solution
    u0 = Expression('t*pow(x[0],2)*(0.5 - x[0]/3)', t=0)

    # Stopping time
    T = 1.0

    # numerical parameters
    h_values = [0.5*2**(-i) for i in range(9)]
    E_values = []
    for h in h_values:
        # mesh parameters
        Nx = int(round(1./np.sqrt(h)))
        Ny = int(round(1./np.sqrt(h)))

        # update source term and time step
        dt = h
        f.dt = dt

        # make function space
        V, mesh = function_space(1, 1, Nx)

        # set t=0 for time-dependent terms before solving
        u0.t = 0
        f.t = 0

        # solve
        u = solver(f, alpha, V, rho, dt, T, u0)

        # Compute error
        u0.t = T
        u_e = interpolate(u0, V)
        e = u_e.vector().array() - u.vector().array()
        E = np.sqrt(np.sum(e**2)/u.vector().array().size)
        E_values.append(E)

    # Compute convergence rate
    r = [np.log(E_values[i-1]/E_values[i])/
         np.log(h_values[i-1]/h_values[i])
         for i in range(len(h_values))]
    # Round to two decimals
    r = [round(r_, 2) for r_ in r]
    
    print 'E: %s' % E_values
    print 'r: %s' % r


def gaussian():
    """Nonlinear diffusion of Gaussian"""

    beta = 2.0
    # diffusion coeffiecent
    alpha = lambda u: 1 + beta*u**2

    rho = Constant(1.0)
    f = Constant(0.0)	# source term

    sigma = 0.3		# width of Gaussian
    # initial condition
    I = Expression('exp(-(x[0]*x[0] + x[1]*x[1])/(2*sigma*sigma))', sigma=sigma)

    # numerical parameters
    dt = 0.01
    dx = 0.05; dy = 0.05
    Nx = int(round(1./dx))
    Ny = int(round(1./dy))
    T = 0.5

    # make function space and solve
    V, mesh = function_space(1, 2, Nx, Ny) 
    u = solver(f, alpha, V, rho, dt, T, u0=I, animate=True)



if __name__ == '__main__':
    'd) run py.test diffusion.py to test'
    #test_constant()

    'e)'
    #error_measure()

    'f)'
    #manufactured_solution()

    'h)'
    #convergence_rate()

    'i)'
    #gaussian()


    
    








