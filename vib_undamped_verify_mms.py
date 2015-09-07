import sympy as sym
import numpy as np
import matplotlib.pyplot as plt

V, t, I, w, dt = sym.symbols('V t I w dt')  # global symbols
f = None  # global variable for the source term in the ODE

def ode_source_term(u):
    """Return the terms in the ODE that the source term
    must balance, here u'' + w**2*u.
    u is symbolic Python function of t."""
    return sym.diff(u(t), t, t) + w**2*u(t)

def residual_discrete_eq(u):
    """Return the residual of the discrete eq. with u inserted."""
    R = DtDt(u, dt) + w**2*u(t) - f
    return sym.simplify(R)

def residual_discrete_eq_step1(u):
    """Return the residual of the discrete eq. at the first
    step with u inserted."""
    R = residual_discrete_eq(u).subs(t,0)
    return sym.simplify(R)

def DtDt(u, dt):
    """Return 2nd-order finite difference for u_tt.
    u is a symbolic Python function of t.
    """
    return (u(t+dt) - 2*u(t) + u(t-dt))/dt**2

def main(u):
    """
    Given some chosen solution u (as a function of t, implemented
    as a Python function), use the method of manufactured solutions
    to compute the source term f, and check if u also solves
    the discrete equations.
    """
    print '=== Testing exact solution: u(t) = %s ===' % u(t)
    print "Initial conditions u(0) = %s, u'(0) = %s:" % \
          (u(t).subs(t, 0), sym.diff(u(t), t).subs(t, 0))

    # Method of manufactured solution requires fitting f
    global f  # source term in the ODE
    f = sym.simplify(ode_source_term(u))

    # Residual in discrete equations (should be 0)
    print 'residual step1:', residual_discrete_eq_step1(u)
    print 'residual:', residual_discrete_eq(u)

def linear():
    c = d = 1
    main(lambda t: c*t + d)

def quadratic():
    b = c = d = 1
    main(lambda t: b*t**2 + c*t + d)

def cubic():
    a = b = c = d = 1
    main(lambda t: a*t**3 + b*t**2 + c*t + d)


def solver(I1, omega, dt1, T1, V1):
    """
    Solve u'' + w**2*u = f(t) for t in (0,T], u(0)=I and u'(0)=V,
    by a central finite difference method with time step dt.
    """
    dt1 = float(dt1)
    Nt = int(round(T1/dt1))
    un = np.zeros(Nt+1)
    tlist = np.linspace(0, T1, Nt+1)

    un[0] = I1
    # f is source term, equal to lhs of Ode
    fi = float(f.subs([(w, omega), (t, tlist[0])]))
    
    # first time step
    un[1] = un[0] - 0.5*dt1**2*omega**2*un[0] + dt1*V1 + 0.5*dt1**2*fi
    for n in range(1, Nt):
        fi = float(f.subs([(w, omega), (t, tlist[n])]))
        un[n+1] = 2*un[n] - un[n-1] + dt1**2*(fi - omega**2*un[n])
    return tlist, un

if __name__ == '__main__':
    #linear()
    quadratic()
    #cubic()

    I1 = 1
    omega = 1
    dt1 = 0.01
    T1 = 2
    V1 = 1
    tlist, un = solver(I1, omega, dt1, T1, V1)
    
    u_exact = tlist**2 + tlist + 1

    plt.plot(tlist, un, 'b-', tlist, u_exact, 'g-')
    plt.xlabel('t')
    plt.ylabel('u')
    plt.legend(['Numerical', 'Exact'])
    plt.title('Verify numerical method using mms')
    plt.show()

    error = abs(u_exact - un)
    plt.plot(tlist, error)
    plt.xlabel('t')
    plt.ylabel('Error')
    plt.title('Absolute value of error between numerical and exact solution')
    plt.show()



"""
Output when run as
python vib_undamped_verify_mms.py
with quadratic manufactured solution:

=== Testing exact solution: u(t) = t**2 + t + 1 ===
Initial conditions u(0) = 1, u'(0) = 1:
residual step1: 0
residual: 0

- plot of numerical and exact solution
- error plot
"""





 
