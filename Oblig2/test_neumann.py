#!/usr/bin/env python
"""
1D wave equation with Neumann conditions
and variable wave velocity::

 u, x, t, cpu = solver(I, V, f, c, U_0, U_L, L, dt, C, T,
                       user_action=None, version='scalar',
                       stability_safety_factor=1.0)

Solve the wave equation u_tt = (c**2*u_x)_x + f(x,t) on (0,L) with
du/dn=0 on x=0 and x=L.
Initial conditions: u=I(x), u_t=V(x).

T is the stop time for the simulation.
dt is the desired time step.
C is the Courant number (=max(c)*dt/dx).
stability_safety_factor enters the stability criterion:
C <= stability_safety_factor (<=1).

I, f, V and c are functions: I(x), f(x,t), V(x), c(x).
f and V can also be 0 or None (equivalent to 0). 
c can be a number or a function c(x).

user_action is a function of (u, x, t, n) where the calling code
can add visualization, error computations, data analysis,
store solutions, etc.
"""
import time, glob, shutil, os
import numpy as np

def solver(I, V, f, c, L, dt, C, T,
           user_action=None,
           stability_safety_factor=1.0):
    """Solve u_tt=(c^2*u_x)_x + f on (0,L)x(0,T]."""
    Nt = int(round(T/dt))
    t = np.linspace(0, Nt*dt, Nt+1)      # Mesh points in time

    # Find max(c) using a fake mesh and adapt dx to C and dt
    if isinstance(c, (float,int)):
        c_max = c
    elif callable(c):
        c_max = max([c(x_) for x_ in np.linspace(0, L, 101)])
    dx = float((dt*c_max)/C)  

    Nx = int(round(L/dx))
    x = np.linspace(0, L, Nx+1)          # Mesh points in space

    # Treat c(x) as array
    if isinstance(c, (float,int)):
        c = np.zeros(x.shape) + c
    elif callable(c):
        # Call c(x) and fill array c
        c_ = np.zeros(x.shape)
        for i in range(Nx+1):
            c_[i] = c(x[i])
        c = c_

    q = c**2
    C2 = (dt/dx)**2; dt2 = dt*dt    # Help variables in the scheme

    # Wrap user-given f, I, V if None or 0
    if f is None or f == 0:
        f = (lambda x, t: 0)
    if I is None or I == 0:
        I = (lambda x: 0)
    if V is None or V == 0:
        V = (lambda x: 0)

    u   = np.zeros(Nx+1)   # Solution array at new time level
    u_1 = np.zeros(Nx+1)   # Solution at 1 time level back
    u_2 = np.zeros(Nx+1)   # Solution at 2 time levels back

    import time;  t0 = time.clock()  # CPU time measurement

    Ix = range(0, Nx+1)
    It = range(0, Nt+1)

    # Load initial condition into u_1
    for i in range(0, Nx+1):
        u_1[i] = I(x[i])

    if user_action is not None:
        user_action(u_1, x, t, 0)

    # Special formula for the first step
    for i in Ix[1:-1]:
        u[i] = u_1[i] + dt*V(x[i]) + \
        0.5*C2*(0.5*(q[i] + q[i+1])*(u_1[i+1] - u_1[i]) - \
                0.5*(q[i] + q[i-1])*(u_1[i] - u_1[i-1])) + \
        0.5*dt2*f(x[i], t[0])

    i = Ix[0]
    # Set boundary values (x=0: i-1 -> i+1 since u[i-1]=u[i+1]
    # when du/dn = 0, on x=L: i+1 -> i-1 since u[i+1]=u[i-1])
    ip1 = i+1
    im1 = ip1  # i-1 -> i+1
    u[i] = u_1[i] + dt*V(x[i]) + \
               C2*0.5*(q[i] + q[ip1])*(u_1[im1] - u_1[i]) + \
               0.5*dt2*f(x[i], t[0])

    i = Ix[-1]
    im1 = i-1
    ip1 = im1  # i+1 -> i-1
    u[i] = u_1[i] + dt*V(x[i]) + \
               C2*0.5*(q[i] + q[ip1])*(u_1[im1] - u_1[i]) + \
               0.5*dt2*f(x[i], t[0])

    if user_action is not None:
        user_action(u, x, t, 1)

    # Update data structures for next step
    #u_2[:] = u_1;  u_1[:] = u  # safe, but slower
    u_2, u_1, u = u_1, u, u_2

    for n in It[1:-1]:
        # Update all inner points
        for i in Ix[1:-1]:
            u[i] = - u_2[i] + 2*u_1[i] + \
                C2*(0.5*(q[i] + q[i+1])*(u_1[i+1] - u_1[i])  - \
                    0.5*(q[i] + q[i-1])*(u_1[i] - u_1[i-1])) + \
                dt2*f(x[i], t[n])



        # Insert boundary conditions
        i = Ix[0]
        # Set boundary values
        # x=0: i-1 -> i+1 since u[i-1]=u[i+1] when du/dn=0
        # x=L: i+1 -> i-1 since u[i+1]=u[i-1] when du/dn=0
        ip1 = i+1
        im1 = ip1
        u[i] = - u_2[i] + 2*u_1[i] + \
               C2*(q[i] + q[im1])*(u_1[im1] - u_1[i])  + \
               dt2*f(x[i], t[n])

        i = Ix[-1]
        im1 = i-1
        ip1 = im1
        u[i] = - u_2[i] + 2*u_1[i] + \
               C2*(q[i] + q[ip1])*(u_1[ip1] - u_1[i])  + \
               dt2*f(x[i], t[n])

        if user_action is not None:
            if user_action(u, x, t, n+1):
                break

        # Update data structures for next step
        #u_2[:] = u_1;  u_1[:] = u  # safe, but slower
        u_2, u_1, u = u_1, u, u_2

    # Important to correct the mathematically wrong u=u_2 above
    # before returning u
    u = u_1
    cpu_time = t0 - time.clock()  
    return u, x, t, cpu_time

def viz(I, V, f, c, L, dt, C, T, umin, umax, u_exact, user):
    """Run solver and visualize u at each time level."""
    import scitools.std as plt
    import time, glob, os

    def plot_u(u, x, t, n):
        """user_action function for solver."""
        u_exact_n = u_exact(x, t[n])
        plt.plot(x, u, 'r-', x, u_exact_n, 'b-',
                 xlabel='x', ylabel='u',
                 axis=[0, L, umin, umax],
                 title='t=%f' % t[n], show=True)
        # Let the initial condition stay on the screen for 2
        # seconds, else insert a pause of 0.2 s between each plot
        time.sleep(2) if t[n] == 0 else time.sleep(0.002)
        plt.savefig('frame_%04d.png' % n)  # for movie making

    def plot_error(u, x, t, n):
        """user_action function for solver.""" 
        u_exact_n = u_exact(x, t[n])
        error = abs(u_exact_n - u)
        plt.plot(x, error, 'r-',
                 xlabel='x', ylabel='Error',
                 axis=[0, L, umin, umax],
                 title='t=%f' % t[n], show=True)
        # Let the initial condition stay on the screen for 2
        # seconds, else insert a pause of 0.2 s between each plot
        time.sleep(2) if t[n] == 0 else time.sleep(0.002)
        plt.savefig('frame_%04d.png' % n)  # for movie making
    
    def convergence_rate(u, x, t, n):
        """user_action function for solver."""
        u_exact_n = u_exact(x, t[n])
        # store error squared at time tn for the whole mesh
        E2 = sum((u_exact_n - u)**2)
        outfile.write('%f' % E2 + '\n')

    if user == 'animate':
        user_action = plot_u
    elif user == 'error_plot':
        user_action = plot_error
    elif user == 'convergence':
        user_action = convergence_rate
        outfile = open('error.dat', 'w')


        

    # Clean up old movie frames
    for filename in glob.glob('frame_*.png'):
        os.remove(filename)


        
    u, x, t, cpu = solver(I, V, f, c, L, dt, C, T, user_action)

    if user_action == 'animate' or user_action == 'error_plot':
        # Make movie files
        fps = 50  # Frames per second
        plt.movie('frame_*.png', encoder='html', fps=fps,
                  output_file='movie.html')
        codec2ext = dict(flv='flv', libx264='mp4', libvpx='webm',
                         libtheora='ogg')
        filespec = 'frame_%04d.png'
        movie_program = 'avconv'  # or 'ffmpeg'
        for codec in codec2ext:
            ext = codec2ext[codec]
            cmd = '%(movie_program)s -r %(fps)d -i %(filespec)s '\
                  '-vcodec %(codec)s movie.%(ext)s' % vars()
            os.system(cmd)

    return u, x, t, cpu

import sympy as sym
from math import *

def source_term(L):
    """
    Adapt source term to exact solution u_exact using sympy
    """
    un, xn, tn, fn, w, qn, Ln = sym.symbols('un xn tn fn w qn Ln')
    #qn = 1 + (xn-(Ln/2))**4		# a)
    qn = 1 + sym.cos(sym.pi*xn/Ln)	# b)
    w = 1
    un = sym.cos(sym.pi*xn/Ln)*sym.cos(w*tn)
    fn = un.diff(tn, 2) - (qn*un.diff(xn)).diff(xn)
    # convert to callable python lambda function
    f = sym.lambdify((xn, tn), sym.simplify(fn.subs(Ln,L)), "math")
    return f
 
def convergence_rates():
    """Constructing the simulation and computing convergence rates"""

    L = 1.					# mesh from x=0 to x=L						
    c = lambda x: np.sqrt(1 + np.cos(np.pi*x/L))# wave speed b)
    #c = lambda x: np.sqrt(1 + (x-L/2)**4)	# wave speed a)
    w = 1.					# angular frequency
    A = 1.				        # amplitude
    num_periods = 2.
    T = 2*np.pi*num_periods
    #dt = L/100.				# time step
    C = 0.9					# Courant number

    # initial contion u(x,0) = I(x)
    I = lambda x: A*np.cos(np.pi*x/L)

    # exact solution
    u_exact = lambda x, t: A*np.cos(np.pi*x/L)*np.cos(w*t)

    umin = -1.2*A;  umax = -umin
    f = source_term(L)
    
    # animate
    #u, x, t, cpu = viz(I, 0, f, c, L, dt, C, T, umin, umax, u_exact, user='animate')
    
    # compute convergence rate for experiments with deacreasing time step
    dt_values = [0.1*2**(-i) for i in range(7)]
    E_values = []
    for dt in dt_values:
        u, x, t, cpu = viz(I, 0, f, c, L, dt, C, T, umin, umax, u_exact, user='convergence')
        infile = open('error.dat', 'r')
        E = 0
        for line in infile:
           E += float(line)
        dx = x[1]
        E = np.sqrt(dx*dt*E)
        E_values.append(E)
    print E_values
    print dt_values
    r = compute_rates(dt_values, E_values)
    print 'r: %s' % r
    
    
    

def compute_rates(dt_values, E_values):
    """compute convergence rates"""
    m = len(dt_values)
    r = [np.log(E_values[i-1]/E_values[i])/
         np.log(dt_values[i-1]/dt_values[i])
         for i in range(0, m, 1)]
    # Round to two decimals
    r = [round(r_, 2) for r_ in r]
    return r

convergence_rates()  



