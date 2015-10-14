#!/usr/bin/env python
"""
2D wave equation solved by finite differences::

  dt, cpu_time = solver(I, V, f, c, Lx, Ly, Nx, Ny, dt, T,
                        user_action=None, version='scalar',
                        stability_safety_factor=1)

Solve the 2D wave equation u_tt = u_xx + u_yy + f(x,t) on (0,L) with
u=0 on the boundary and initial condition du/dt=0.

Nx and Ny are the total number of mesh cells in the x and y
directions. The mesh points are numbered as (0,0), (1,0), (2,0),
..., (Nx,0), (0,1), (1,1), ..., (Nx, Ny).

dt is the time step. If dt<=0, an optimal time step is used.
T is the stop time for the simulation.

I, V, f are functions: I(x,y), V(x,y), f(x,y,t). V and f
can be specified as None or 0, resulting in V=0 and f=0.

user_action: function of (u, x, y, t, n) called at each time
level (x and y are one-dimensional coordinate vectors).
This function allows the calling code to plot the solution,
compute errors, etc.
"""
import time, sys
#from scitools.std import *
from numpy import *

def solver(I, V, f, c, Lx, Ly, Nx, Ny, dt, T, b,
           user_action=None, version='scalar',
           safety_factor=1.0, C=1.0, dim=1):

    if version == 'scalar':
        advance = advance_scalar
    elif version == 'vectorized':
        advance = advance_vectorized
    else:
        raise ValueError('version=%s' % version)

    x = linspace(0, Lx, Nx+1)  # mesh points in x dir
    y = linspace(0, Ly, Ny+1)  # mesh points in y dir
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    xv = x[:,newaxis]         # for vectorized function evaluations
    yv = y[newaxis,:]
    
    if isinstance(c, (float,int)):
        c_max = c
    elif callable(c):
        c_max = (c(xv,yv)).max()

    if dim == 2:
        stability_limit = (1./c_max)*(1./(sqrt((1./dx**2) + (1./dy**2))))
        print stability_limit
    else:
        stability_limit = (dx*safety_factor*C)/c_max
    
    if dt <= 0:                # max time step?
        safety_factor = -dt    # use negative dt as safety factor
        dt = safety_factor*stability_limit
    elif dt > stability_limit:
        print 'error: dt=%g exceeds the stability limit %g' % \
              (dt, stability_limit)
    Nt = int(round(T/float(dt)))
    t = linspace(0, T, Nt+1)    # mesh points in time

    # Treat c(x) as array
    if isinstance(c, (float,int)):
        c = zeros((Nx+1,Ny+1)) + c
    elif callable(c):
        # Call c(x) and fill array c
        c_ = zeros((Nx+1,Ny+1))
        c_[:,:] = c(xv, yv)
        c = c_
  
    q = c**2		# help variable


    Cx2 = (dt/dx)**2;  Cy2 = (dt/dy)**2    # help variables
    dt2 = dt**2
    D = 0.5*b*dt

    # Allow f and V to be None or 0
    if f is None or f == 0:
        f = (lambda x, y, t: 0) if version == 'scalar' else \
            lambda x, y, t: zeros((Nx+1, Ny+1))
        # or simpler: x*y*0
    if V is None or V == 0:
        V = (lambda x, y: 0) if version == 'scalar' else \
            lambda x, y: zeros((Nx+1, Ny+1))


    
    u   = zeros((Nx+1,Ny+1))   # solution array
    u_1 = zeros((Nx+1,Ny+1))   # solution at t-dt
    u_2 = zeros((Nx+1,Ny+1))   # solution at t-2*dt
    f_a = zeros((Nx+1,Ny+1))   # for compiled loops

    Ix = range(0, u.shape[0])
    Iy = range(0, u.shape[1])
    It = range(0, t.shape[0])

    import time; t0 = time.clock()          # for measuring CPU time

    # Load initial condition into u_1
    if version == 'scalar' or version == 'vectorized':
        for i in Ix:
            for j in Iy:
                u_1[i,j] = I(x[i], y[j])
    else: # use vectorized version
        u_1[:,:] = I(xv, yv)

    
    if user_action is not None:
        user_action(u_1, x, xv, y, yv, t, 0)

    # Special formula for first time step
    n = 0
    # First step requires a special formula, use either the scalar
    # or vectorized version (the impact of more efficient loops than
    # in advance_vectorized is small as this is only one step)
    if version == 'scalar':
        for i in Ix[1:-1]:
            for j in Iy[1:-1]:
                dqdx = 0.5*(q[i,j] + q[i+1,j])*(u_1[i+1,j] - u_1[i,j]) - \
                       0.5*(q[i,j] + q[i-1,j])*(u_1[i,j] - u_1[i-1,j])
                dqdy = 0.5*(q[i,j] + q[i,j+1])*(u_1[i,j+1] - u_1[i,j]) - \
                       0.5*(q[i,j] + q[i,j-1])*(u_1[i,j] - u_1[i,j-1])
                u[i,j] = u_1[i,j] - 0.5*dt*(b*dt-2)*V(x[i], y[j]) + \
                         0.5*Cx2*dqdx + 0.5*Cy2*dqdy + 0.5*dt2*f(x[i], y[j], t[n])   

        # boundary coditions

	# (0,j)
        i = Ix[0]
        ip1 = i+1
        im1 = ip1
        for j in Iy[1:-1]:
            dqdx = 0.5*(q[i,j] + q[ip1,j])*(u_1[ip1,j] - u_1[i,j]) - \
                   0.5*(q[i,j] + q[im1,j])*(u_1[i,j] - u_1[im1,j])
            dqdy = 0.5*(q[i,j] + q[i,j+1])*(u_1[i,j+1] - u_1[i,j]) - \
                   0.5*(q[i,j] + q[i,j-1])*(u_1[i,j] - u_1[i,j-1])
            u[i,j] = u_1[i,j] - 0.5*dt*(b*dt-2)*V(x[i], y[j]) + \
                     0.5*Cx2*dqdx + 0.5*Cy2*dqdy + 0.5*dt2*f(x[i], y[j], t[n]) 

        # (Lx,j)
        i = Ix[-1]
        im1 = i-1
        ip1 = im1
        for j in Iy[1:-1]:
            dqdx = 0.5*(q[i,j] + q[ip1,j])*(u_1[ip1,j] - u_1[i,j]) - \
                   0.5*(q[i,j] + q[im1,j])*(u_1[i,j] - u_1[im1,j])
            dqdy = 0.5*(q[i,j] + q[i,j+1])*(u_1[i,j+1] - u_1[i,j]) - \
                   0.5*(q[i,j] + q[i,j-1])*(u_1[i,j] - u_1[i,j-1])
            u[i,j] = u_1[i,j] - 0.5*dt*(b*dt-2)*V(x[i], y[j]) + \
                     0.5*Cx2*dqdx + 0.5*Cy2*dqdy + 0.5*dt2*f(x[i], y[j], t[n]) 

        # (i,0)
        j = Iy[0] 
        jp1 = j+1
        jm1 = jp1
        for i in Ix[1:-1]:
            dqdx = 0.5*(q[i,j] + q[i+1,j])*(u_1[i+1,j] - u_1[i,j]) - \
                   0.5*(q[i,j] + q[i-1,j])*(u_1[i,j] - u_1[i-1,j])
            dqdy = 0.5*(q[i,j] + q[i,jp1])*(u_1[i,jp1] - u_1[i,j]) - \
                   0.5*(q[i,j] + q[i,jm1])*(u_1[i,j] - u_1[i,jm1])
            u[i,j] = u_1[i,j] - 0.5*dt*(b*dt-2)*V(x[i], y[j]) + \
                     0.5*Cx2*dqdx + 0.5*Cy2*dqdy + 0.5*dt2*f(x[i], y[j], t[n])   

        # (i,Ly)
        j = Iy[-1]
        jm1 = j-1
        jp1 = jm1     
        for i in Ix[1:-1]:
            dqdx = 0.5*(q[i,j] + q[i+1,j])*(u_1[i+1,j] - u_1[i,j]) - \
                   0.5*(q[i,j] + q[i-1,j])*(u_1[i,j] - u_1[i-1,j])
            dqdy = 0.5*(q[i,j] + q[i,jp1])*(u_1[i,jp1] - u_1[i,j]) - \
                   0.5*(q[i,j] + q[i,jm1])*(u_1[i,j] - u_1[i,jm1])
            u[i,j] = u_1[i,j] - 0.5*dt*(b*dt-2)*V(x[i], y[j]) + \
                     0.5*Cx2*dqdx + 0.5*Cy2*dqdy + 0.5*dt2*f(x[i], y[j], t[n])


    else:
        f_a[:,:] = f(xv, yv, t[n])  # precompute, size as u
        V_a = V(xv, yv)

        dt = sqrt(dt2)  # save
        dqdx = 0.5*(q[1:-1,1:-1] + q[2:,1:-1])*(u_1[2:,1:-1] - u_1[1:-1,1:-1]) - \
               0.5*(q[1:-1,1:-1] + q[:-2,1:-1])*(u_1[1:-1,1:-1] - u_1[:-2,1:-1])
        dqdy = 0.5*(q[1:-1,1:-1] + q[1:-1,2:])*(u_1[1:-1,2:] - u_1[1:-1,1:-1]) - \
               0.5*(q[1:-1,1:-1] + q[1:-1,:-2])*(u_1[1:-1,1:-1] - u_1[1:-1,:-2])
        u[1:-1,1:-1] = u_1[1:-1,1:-1] - 0.5*dt*(b*dt-2)*V_a[1:-1,1:-1] + \
                       0.5*Cx2*dqdx + 0.5*Cy2*dqdy + 0.5*dt2*f_a[1:-1,1:-1]   

        # boundary conditions

        # (0,j)
        i = Ix[0]
        ip1 = i+1
        im1 = ip1
        dqdx = 0.5*(q[i,1:-1] + q[ip1,1:-1])*(u_1[ip1,1:-1] - u_1[i,1:-1]) - \
               0.5*(q[i,1:-1] + q[im1,1:-1])*(u_1[i,1:-1] - u_1[im1,1:-1])
        dqdy = 0.5*(q[i,1:-1] + q[i,2:])*(u_1[i,2:] - u_1[i,1:-1]) - \
               0.5*(q[i,1:-1] + q[i,:-2])*(u_1[i,1:-1] - u_1[i,:-2])
        u[i,1:-1] = u_1[i,1:-1] - 0.5*dt*(b*dt-2)*V_a[i,1:-1] + \
                       0.5*Cx2*dqdx + 0.5*Cy2*dqdy + 0.5*dt2*f_a[i,1:-1]   

        # (Lx,j)
        i = Ix[-1]
        im1 = i-1
        ip1 = im1
        dqdx = 0.5*(q[i,1:-1] + q[ip1,1:-1])*(u_1[ip1,1:-1] - u_1[i,1:-1]) - \
               0.5*(q[i,1:-1] + q[im1,1:-1])*(u_1[i,1:-1] - u_1[im1,1:-1])
        dqdy = 0.5*(q[i,1:-1] + q[i,2:])*(u_1[i,2:] - u_1[i,1:-1]) - \
               0.5*(q[i,1:-1] + q[i,:-2])*(u_1[i,1:-1] - u_1[i,:-2])
        u[i,1:-1] = u_1[i,1:-1] - 0.5*dt*(b*dt-2)*V_a[i,1:-1] + \
                       0.5*Cx2*dqdx + 0.5*Cy2*dqdy + 0.5*dt2*f_a[i,1:-1]

        # (i,0)
        j = Iy[0]
        jp1 = j+1
        jm1 = jp1
        dqdx = 0.5*(q[1:-1,j] + q[2:,j])*(u_1[2:,j] - u_1[1:-1,j]) - \
               0.5*(q[1:-1,j] + q[:-2,j])*(u_1[1:-1,j] - u_1[:-2,j])
        dqdy = 0.5*(q[1:-1,j] + q[1:-1,jp1])*(u_1[1:-1,jp1] - u_1[1:-1,j]) - \
               0.5*(q[1:-1,j] + q[1:-1,jm1])*(u_1[1:-1,j] - u_1[1:-1,jm1])
        u[1:-1,j] = u_1[1:-1,j] - 0.5*dt*(b*dt-2)*V_a[1:-1,j] + \
                       0.5*Cx2*dqdx + 0.5*Cy2*dqdy + 0.5*dt2*f_a[1:-1,j]              


        # (i,Ly)
        j = Iy[-1]
        jm1 = j-1
        jp1 = jm1
        dqdx = 0.5*(q[1:-1,j] + q[2:,j])*(u_1[2:,j] - u_1[1:-1,j]) - \
               0.5*(q[1:-1,j] + q[:-2,j])*(u_1[1:-1,j] - u_1[:-2,j])
        dqdy = 0.5*(q[1:-1,j] + q[1:-1,jp1])*(u_1[1:-1,jp1] - u_1[1:-1,j]) - \
               0.5*(q[1:-1,j] + q[1:-1,jm1])*(u_1[1:-1,j] - u_1[1:-1,jm1])
        u[1:-1,j] = u_1[1:-1,j] - 0.5*dt*(b*dt-2)*V_a[1:-1,j] + \
                       0.5*Cx2*dqdx + 0.5*Cy2*dqdy + 0.5*dt2*f_a[1:-1,j]     


    # corners 
    # (0,0)
    i = Ix[0]
    ip1 = i+1
    im1 = ip1
    j = Iy[0]
    jp1 = j+1
    jm1 = jp1
    dqdx = 0.5*(q[i,j] + q[ip1,j])*(u_1[ip1,j] - u_1[i,j]) - \
           0.5*(q[i,j] + q[im1,j])*(u_1[i,j] - u_1[im1,j])
    dqdy = 0.5*(q[i,j] + q[i,jp1])*(u_1[i,jp1] - u_1[i,j]) - \
           0.5*(q[i,j] + q[i,jm1])*(u_1[i,j] - u_1[i,jm1])
    if version == 'scalar':
        u[i,j] = u_1[i,j] - 0.5*dt*(b*dt-2)*V(x[i], y[j]) + \
                 0.5*Cx2*dqdx + 0.5*Cy2*dqdy + 0.5*dt2*f(x[i], y[j], t[n]) 
    else: 
        u[i,j] = u_1[i,j] - 0.5*dt*(b*dt-2)*V_a[i,j] + \
                 0.5*Cx2*dqdx + 0.5*Cy2*dqdy + 0.5*dt2*f_a[i,j]         

    # (Lx,Ly)
    i = Ix[-1]
    im1 = i-1
    ip1 = im1
    j = Iy[-1]
    jm1 = j-1
    jp1 = jm1
    dqdx = 0.5*(q[i,j] + q[ip1,j])*(u_1[ip1,j] - u_1[i,j]) - \
           0.5*(q[i,j] + q[im1,j])*(u_1[i,j] - u_1[im1,j])
    dqdy = 0.5*(q[i,j] + q[i,jp1])*(u_1[i,jp1] - u_1[i,j]) - \
           0.5*(q[i,j] + q[i,jm1])*(u_1[i,j] - u_1[i,jm1])
    if version == 'scalar':
        u[i,j] = u_1[i,j] - 0.5*dt*(b*dt-2)*V(x[i], y[j]) + \
                 0.5*Cx2*dqdx + 0.5*Cy2*dqdy + 0.5*dt2*f(x[i], y[j], t[n]) 
    else: 
        u[i,j] = u_1[i,j] - 0.5*dt*(b*dt-2)*V_a[i,j] + \
                 0.5*Cx2*dqdx + 0.5*Cy2*dqdy + 0.5*dt2*f_a[i,j]  

    # (0,Ly)
    i = Ix[0]
    ip1 = i+1
    im1 = ip1
    j = Iy[-1]
    jm1 = j-1
    jp1 = jm1
    dqdx = 0.5*(q[i,j] + q[ip1,j])*(u_1[ip1,j] - u_1[i,j]) - \
           0.5*(q[i,j] + q[im1,j])*(u_1[i,j] - u_1[im1,j])
    dqdy = 0.5*(q[i,j] + q[i,jp1])*(u_1[i,jp1] - u_1[i,j]) - \
           0.5*(q[i,j] + q[i,jm1])*(u_1[i,j] - u_1[i,jm1])
    if version == 'scalar':
        u[i,j] = u_1[i,j] - 0.5*dt*(b*dt-2)*V(x[i], y[j]) + \
                 0.5*Cx2*dqdx + 0.5*Cy2*dqdy + 0.5*dt2*f(x[i], y[j], t[n]) 
    else: 
        u[i,j] = u_1[i,j] - 0.5*dt*(b*dt-2)*V_a[i,j] + \
                 0.5*Cx2*dqdx + 0.5*Cy2*dqdy + 0.5*dt2*f_a[i,j]  

    # (Lx,0)
    i = Ix[-1]
    im1 = i-1
    ip1 = im1
    j = Iy[0]
    jp1 = j+1
    jm1 = jp1
    dqdx = 0.5*(q[i,j] + q[ip1,j])*(u_1[ip1,j] - u_1[i,j]) - \
           0.5*(q[i,j] + q[im1,j])*(u_1[i,j] - u_1[im1,j])
    dqdy = 0.5*(q[i,j] + q[i,jp1])*(u_1[i,jp1] - u_1[i,j]) - \
           0.5*(q[i,j] + q[i,jm1])*(u_1[i,j] - u_1[i,jm1])
    if version == 'scalar':
        u[i,j] = u_1[i,j] - 0.5*dt*(b*dt-2)*V(x[i], y[j]) + \
                 0.5*Cx2*dqdx + 0.5*Cy2*dqdy + 0.5*dt2*f(x[i], y[j], t[n]) 
    else: 
        u[i,j] = u_1[i,j] - 0.5*dt*(b*dt-2)*V_a[i,j] + \
                 0.5*Cx2*dqdx + 0.5*Cy2*dqdy + 0.5*dt2*f_a[i,j]  


    if user_action is not None:
        user_action(u, x, xv, y, yv, t, 1)

    # Update data structures for next step
    #u_2[:] = u_1;  u_1[:] = u  # safe, but slower
    u_2, u_1, u = u_1, u, u_2

    for n in It[1:-1]:
        if version == 'scalar':
            # use f(x,y,t) function
            u = advance(u, u_1, u_2, f, x, y, t, n, Cx2, Cy2, dt2, D, q)
            #mesh(x,y,u)
            #raw_input('Press Enter to continue: ')
        
        else:
            f_a[:,:] = f(xv, yv, t[n])  # precompute, size as u
            u = advance(u, u_1, u_2, f_a, Cx2, Cy2, dt2, D, q)

        if user_action is not None:
            if user_action(u, x, xv, y, yv, t, n+1):
                break

        # Update data structures for next step
        #u_2[:] = u_1;  u_1[:] = u  # safe, but slower
        u_2, u_1, u = u_1, u, u_2

    # Important to set u = u_1 if u is to be returned!
    t1 = time.clock()
    # dt might be computed in this function so return the value
    return u, x, xv, y, yv, t, t1 - t0



def advance_scalar(u, u_1, u_2, f, x, y, t, n, Cx2, Cy2, dt2, D, q):
    Ix = range(0, u.shape[0]);  Iy = range(0, u.shape[1])
    for i in Ix[1:-1]:
        for j in Iy[1:-1]:
            dqdx = 0.5*(q[i,j] + q[i+1,j])*(u_1[i+1,j] - u_1[i,j]) - \
                   0.5*(q[i,j] + q[i-1,j])*(u_1[i,j] - u_1[i-1,j])
            dqdy = 0.5*(q[i,j] + q[i,j+1])*(u_1[i,j+1] - u_1[i,j]) - \
                   0.5*(q[i,j] + q[i,j-1])*(u_1[i,j] - u_1[i,j-1])
            u[i,j] = (1./(1+D))*(2*u_1[i,j] + u_2[i,j]*(D-1) +  \
                     Cx2*dqdx + Cy2*dqdy + dt2*f(x[i], y[j], t[n])) 

    # Boundary conditions

    # (0,j)
    i = Ix[0]
    ip1 = i+1
    im1 = ip1
    for j in Iy[1:-1]:
        dqdx = 0.5*(q[i,j] + q[ip1,j])*(u_1[ip1,j] - u_1[i,j]) - \
               0.5*(q[i,j] + q[im1,j])*(u_1[i,j] - u_1[im1,j])
        dqdy = 0.5*(q[i,j] + q[i,j+1])*(u_1[i,j+1] - u_1[i,j]) - \
               0.5*(q[i,j] + q[i,j-1])*(u_1[i,j] - u_1[i,j-1])
        u[i,j] = (1./(1+D))*(2*u_1[i,j] + u_2[i,j]*(D-1) +  \
                 Cx2*dqdx + Cy2*dqdy + dt2*f(x[i], y[j], t[n]))     
 
    # (Lx,j)   
    i = Ix[-1]
    im1 = i-1
    ip1 = im1
    for j in Iy[1:-1]:
        dqdx = 0.5*(q[i,j] + q[ip1,j])*(u_1[ip1,j] - u_1[i,j]) - \
               0.5*(q[i,j] + q[im1,j])*(u_1[i,j] - u_1[im1,j])
        dqdy = 0.5*(q[i,j] + q[i,j+1])*(u_1[i,j+1] - u_1[i,j]) - \
               0.5*(q[i,j] + q[i,j-1])*(u_1[i,j] - u_1[i,j-1])
        u[i,j] = (1./(1+D))*(2*u_1[i,j] + u_2[i,j]*(D-1) +  \
                 Cx2*dqdx + Cy2*dqdy + dt2*f(x[i], y[j], t[n]))  

    # (i,0)
    j = Iy[0]
    jp1 = j+1
    jm1 = jp1
    for i in Ix[1:-1]:
        dqdx = 0.5*(q[i,j] + q[i+1,j])*(u_1[i+1,j] - u_1[i,j]) - \
               0.5*(q[i,j] + q[i-1,j])*(u_1[i,j] - u_1[i-1,j])
        dqdy = 0.5*(q[i,j] + q[i,jp1])*(u_1[i,jp1] - u_1[i,j]) - \
               0.5*(q[i,j] + q[i,jm1])*(u_1[i,j] - u_1[i,jm1])
        u[i,j] = (1./(1+D))*(2*u_1[i,j] + u_2[i,j]*(D-1) +  \
                 Cx2*dqdx + Cy2*dqdy + dt2*f(x[i], y[j], t[n]))     

    # (i,Ly)
    j = Iy[-1]
    jm1 = j-1
    jp1 = jm1
    for i in Ix[1:-1]:
        dqdx = 0.5*(q[i,j] + q[i+1,j])*(u_1[i+1,j] - u_1[i,j]) - \
               0.5*(q[i,j] + q[i-1,j])*(u_1[i,j] - u_1[i-1,j])
        dqdy = 0.5*(q[i,j] + q[i,jp1])*(u_1[i,jp1] - u_1[i,j]) - \
               0.5*(q[i,j] + q[i,jm1])*(u_1[i,j] - u_1[i,jm1])
        u[i,j] = (1./(1+D))*(2*u_1[i,j] + u_2[i,j]*(D-1) +  \
                 Cx2*dqdx + Cy2*dqdy + dt2*f(x[i], y[j], t[n]))   

    # (0,0)
    i = Ix[0]
    ip1 = i+1
    im1 = ip1
    j = Iy[0]
    jp1 = i+1
    jm1 = jp1
    dqdx = 0.5*(q[i,j] + q[ip1,j])*(u_1[ip1,j] - u_1[i,j]) - \
           0.5*(q[i,j] + q[im1,j])*(u_1[i,j] - u_1[im1,j])
    dqdy = 0.5*(q[i,j] + q[i,jp1])*(u_1[i,jp1] - u_1[i,j]) - \
           0.5*(q[i,j] + q[i,jm1])*(u_1[i,j] - u_1[i,jm1])
    u[i,j] = (1./(1+D))*(2*u_1[i,j] + u_2[i,j]*(D-1) +  \
             Cx2*dqdx + Cy2*dqdy + dt2*f(x[i], y[j], t[n])) 

    # (Lx,Ly)
    i = Ix[-1]
    im1 = i-1
    ip1 = im1
    j = Iy[-1]
    jm1 = j-1
    jp1 = jm1
    dqdx = 0.5*(q[i,j] + q[ip1,j])*(u_1[ip1,j] - u_1[i,j]) - \
           0.5*(q[i,j] + q[im1,j])*(u_1[i,j] - u_1[im1,j])
    dqdy = 0.5*(q[i,j] + q[i,jp1])*(u_1[i,jp1] - u_1[i,j]) - \
           0.5*(q[i,j] + q[i,jm1])*(u_1[i,j] - u_1[i,jm1])
    u[i,j] = (1./(1+D))*(2*u_1[i,j] + u_2[i,j]*(D-1) +  \
             Cx2*dqdx + Cy2*dqdy + dt2*f(x[i], y[j], t[n])) 

    # (0,Ly)
    i = Ix[0]
    ip1 = i+1
    im1 = ip1
    j = Iy[-1]
    jm1 = j-1
    jp1 = jm1
    dqdx = 0.5*(q[i,j] + q[ip1,j])*(u_1[ip1,j] - u_1[i,j]) - \
           0.5*(q[i,j] + q[im1,j])*(u_1[i,j] - u_1[im1,j])
    dqdy = 0.5*(q[i,j] + q[i,jp1])*(u_1[i,jp1] - u_1[i,j]) - \
           0.5*(q[i,j] + q[i,jm1])*(u_1[i,j] - u_1[i,jm1])
    u[i,j] = (1./(1+D))*(2*u_1[i,j] + u_2[i,j]*(D-1) +  \
             Cx2*dqdx + Cy2*dqdy + dt2*f(x[i], y[j], t[n])) 

    # (Lx,0)
    i = Ix[-1]
    im1 = i-1
    ip1 = im1
    j = Iy[0]
    jp1 = j+1
    jm1 = jp1
    dqdx = 0.5*(q[i,j] + q[ip1,j])*(u_1[ip1,j] - u_1[i,j]) - \
           0.5*(q[i,j] + q[im1,j])*(u_1[i,j] - u_1[im1,j])
    dqdy = 0.5*(q[i,j] + q[i,jp1])*(u_1[i,jp1] - u_1[i,j]) - \
           0.5*(q[i,j] + q[i,jm1])*(u_1[i,j] - u_1[i,jm1])
    u[i,j] = (1./(1+D))*(2*u_1[i,j] + u_2[i,j]*(D-1) +  \
             Cx2*dqdx + Cy2*dqdy + dt2*f(x[i], y[j], t[n])) 
 
    return u      


def advance_vectorized(u, u_1, u_2, f_a, Cx2, Cy2, dt2, D, q):
    Ix = range(0, u.shape[0]);  Iy = range(0, u.shape[1])

    dt = sqrt(dt2)  # save
    dqdx = 0.5*(q[1:-1,1:-1] + q[2:,1:-1])*(u_1[2:,1:-1] - u_1[1:-1,1:-1]) - \
           0.5*(q[1:-1,1:-1] + q[:-2,1:-1])*(u_1[1:-1,1:-1] - u_1[:-2,1:-1])
    dqdy = 0.5*(q[1:-1,1:-1] + q[1:-1,2:])*(u_1[1:-1,2:] - u_1[1:-1,1:-1]) - \
           0.5*(q[1:-1,1:-1] + q[1:-1,:-2])*(u_1[1:-1,1:-1] - u_1[1:-1,:-2])
    u[1:-1,1:-1] = (1./(1+D))*(2*u_1[1:-1,1:-1] + u_2[1:-1,1:-1]*(D-1) +  \
             Cx2*dqdx + Cy2*dqdy + dt2*f_a[1:-1,1:-1])

    # Boundary conditions

    # (0,j)
    i = Ix[0]
    ip1 = i+1
    im1 = ip1
    dqdx = 0.5*(q[i,1:-1] + q[ip1,1:-1])*(u_1[ip1,1:-1] - u_1[i,1:-1]) - \
           0.5*(q[i,1:-1] + q[im1,1:-1])*(u_1[i,1:-1] - u_1[im1,1:-1])
    dqdy = 0.5*(q[i,1:-1] + q[i,2:])*(u_1[i,2:] - u_1[i,1:-1]) - \
           0.5*(q[i,1:-1] + q[i,:-2])*(u_1[i,1:-1] - u_1[i,:-2])
    u[i,1:-1] = (1./(1+D))*(2*u_1[i,1:-1] + u_2[i,1:-1]*(D-1) +  \
             Cx2*dqdx + Cy2*dqdy + dt2*f_a[i,1:-1] )

    # (Lx,j)
    i = Ix[-1]
    im1 = i-1
    ip1 = im1
    dqdx = 0.5*(q[i,1:-1] + q[ip1,1:-1])*(u_1[ip1,1:-1] - u_1[i,1:-1]) - \
           0.5*(q[i,1:-1] + q[im1,1:-1])*(u_1[i,1:-1] - u_1[im1,1:-1])
    dqdy = 0.5*(q[i,1:-1] + q[i,2:])*(u_1[i,2:] - u_1[i,1:-1]) - \
           0.5*(q[i,1:-1] + q[i,:-2])*(u_1[i,1:-1] - u_1[i,:-2])
    u[i,1:-1] = (1./(1+D))*(2*u_1[i,1:-1] + u_2[i,1:-1]*(D-1) +  \
             Cx2*dqdx + Cy2*dqdy + dt2*f_a[i,1:-1])

    # (i,0)
    j = Iy[0]
    jp1 = j+1
    jm1 = jp1
    dqdx = 0.5*(q[1:-1,j] + q[2:,j])*(u_1[2:,j] - u_1[1:-1,j]) - \
           0.5*(q[1:-1,j] + q[:-2,j])*(u_1[1:-1,j] - u_1[:-2,j])
    dqdy = 0.5*(q[1:-1,j] + q[1:-1,jp1])*(u_1[1:-1,jp1] - u_1[1:-1,j]) - \
           0.5*(q[1:-1,j] + q[1:-1,jm1])*(u_1[1:-1,j] - u_1[1:-1,jm1])
    u[1:-1,j] = (1./(1+D))*(2*u_1[1:-1,j] + u_2[1:-1,j]*(D-1) +  \
             Cx2*dqdx + Cy2*dqdy + dt2*f_a[1:-1,j])   

    # (i,Ly)
    j = Iy[-1]
    jm1 = j-1
    jp1 = jm1
    dqdx = 0.5*(q[1:-1,j] + q[2:,j])*(u_1[2:,j] - u_1[1:-1,j]) - \
           0.5*(q[1:-1,j] + q[:-2,j])*(u_1[1:-1,j] - u_1[:-2,j])
    dqdy = 0.5*(q[1:-1,j] + q[1:-1,jp1])*(u_1[1:-1,jp1] - u_1[1:-1,j]) - \
           0.5*(q[1:-1,j] + q[1:-1,jm1])*(u_1[1:-1,j] - u_1[1:-1,jm1])
    u[1:-1,j] = (1./(1+D))*(2*u_1[1:-1,j] + u_2[1:-1,j]*(D-1) +  \
             Cx2*dqdx + Cy2*dqdy + dt2*f_a[1:-1,j])
    

    # (0,0)
    i = Ix[0]
    ip1 = i+1
    im1 = ip1
    j = Iy[0]
    jp1 = i+1
    jm1 = jp1
    dqdx = 0.5*(q[i,j] + q[ip1,j])*(u_1[ip1,j] - u_1[i,j]) - \
           0.5*(q[i,j] + q[im1,j])*(u_1[i,j] - u_1[im1,j])
    dqdy = 0.5*(q[i,j] + q[i,jp1])*(u_1[i,jp1] - u_1[i,j]) - \
           0.5*(q[i,j] + q[i,jm1])*(u_1[i,j] - u_1[i,jm1])
    u[i,j] = (1./(1+D))*(2*u_1[i,j] + u_2[i,j]*(D-1) +  \
             Cx2*dqdx + Cy2*dqdy + dt2*f_a[i,j]) 

    # (Lx,Ly)
    i = Ix[-1]
    im1 = i-1
    ip1 = im1
    j = Iy[-1]
    jm1 = j-1
    jp1 = jm1
    dqdx = 0.5*(q[i,j] + q[ip1,j])*(u_1[ip1,j] - u_1[i,j]) - \
           0.5*(q[i,j] + q[im1,j])*(u_1[i,j] - u_1[im1,j])
    dqdy = 0.5*(q[i,j] + q[i,jp1])*(u_1[i,jp1] - u_1[i,j]) - \
           0.5*(q[i,j] + q[i,jm1])*(u_1[i,j] - u_1[i,jm1])
    u[i,j] = (1./(1+D))*(2*u_1[i,j] + u_2[i,j]*(D-1) +  \
             Cx2*dqdx + Cy2*dqdy + dt2*f_a[i,j]) 

    # (0,Ly)
    i = Ix[0]
    ip1 = i+1
    im1 = ip1
    j = Iy[-1]
    jm1 = j-1
    jp1 = jm1
    dqdx = 0.5*(q[i,j] + q[ip1,j])*(u_1[ip1,j] - u_1[i,j]) - \
           0.5*(q[i,j] + q[im1,j])*(u_1[i,j] - u_1[im1,j])
    dqdy = 0.5*(q[i,j] + q[i,jp1])*(u_1[i,jp1] - u_1[i,j]) - \
           0.5*(q[i,j] + q[i,jm1])*(u_1[i,j] - u_1[i,jm1])
    u[i,j] = (1./(1+D))*(2*u_1[i,j] + u_2[i,j]*(D-1) +  \
             Cx2*dqdx + Cy2*dqdy + dt2*f_a[i,j]) 

    # (Lx,0)
    i = Ix[-1]
    im1 = i-1
    ip1 = im1
    j = Iy[0]
    jp1 = j+1
    jm1 = jp1
    dqdx = 0.5*(q[i,j] + q[ip1,j])*(u_1[ip1,j] - u_1[i,j]) - \
           0.5*(q[i,j] + q[im1,j])*(u_1[i,j] - u_1[im1,j])
    dqdy = 0.5*(q[i,j] + q[i,jp1])*(u_1[i,jp1] - u_1[i,j]) - \
           0.5*(q[i,j] + q[i,jm1])*(u_1[i,j] - u_1[i,jm1])
    u[i,j] = (1./(1+D))*(2*u_1[i,j] + u_2[i,j]*(D-1) +  \
             Cx2*dqdx + Cy2*dqdy + dt2*f_a[i,j]) 

    return u


def visualize(I, V, f, c, Lx, Ly, Nx, Ny, dt, T, b, 
              version, safety_factor=1., C=1.,
              u_exact=None, B=None):

    from mpl_toolkits.mplot3d import axes3d
    import matplotlib.pyplot as plt
    import time

    plt.ion()

    if u_exact is not None or B is not None:
        #fig = plt.figure(figsize=plt.figaspect(0.5))
        fig = plt.figure()
    else:
        fig = plt.figure()
        ax = axes3d.Axes3D(fig)
        global wframe
        wframe = None



    def plot_u(u, x, xv, y, yv, t, n):
        X, Y = meshgrid(x, y)       
      
        #ax.set_zlim3d(-10, 10)
        global wframe
        oldcol = wframe   
        
        wframe = ax.plot_wireframe(X, Y, u)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('u')
        ax.set_title('u, t=%.1f' % t[n])


        # Remove old line collection before drawing
        if oldcol is not None:
            ax.collections.remove(oldcol)

        plt.draw()
        #time.sleep(0.01)
   
    # plot u and u_exact 
    
    def plot_u_exact(u, x, xv, y, yv, t, n):
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        X, Y = meshgrid(x, y)

        wframe1 = ax.plot_wireframe(X, Y, u)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('u')
        ax.set_zlim3d(-1, 1)
        ax.set_title('u, t=%.1f' % t[n])

        ax = fig.add_subplot(1, 2, 2, projection='3d')
        X, Y = meshgrid(x, y)
        u_e = u_exact(xv, yv, t[n])

        wframe2 = ax.plot_wireframe(X, Y, u_e)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('u')
        ax.set_zlim3d(-1, 1)
        ax.set_title('u_exact, t=%.1f' % t[n])

        plt.draw()
        #time.sleep(0.02)

    def plot_bottom(u, x, xv, y, yv, t, n):
        ax = fig.add_subplot(111, projection='3d')
        X, Y = meshgrid(x, y)

        ax.plot_wireframe(X, Y, u)

        u_e = B(xv, yv)
        ax.plot_wireframe(X, Y, u_e)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('u')
        ax.set_zlim3d(-0.2, 1.9)
        ax.set_title('t=%.1f' % t[n])
        plt.savefig('waveplot_%04d.png' % (n))
        plt.draw()
        
              
    if u_exact is not None:
        u, x, xv, y, yv, t, cpu = solver(I=I, V=V, f=f, c=c, Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny,
                                         dt=dt, T=T, b=b,
                                         user_action=plot_u_exact, version=version,
                                         safety_factor=safety_factor, C=C)
    elif B is not None:
        u, x, xv, y, yv, t, cpu = solver(I=I, V=V, f=f, c=c, Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny,
                                         dt=dt, T=T, b=b,
                                         user_action=plot_bottom, version=version,
                                         safety_factor=safety_factor, C=C, dim=2)
        
    else:
        u, x, xv, y, yv, t, cpu = solver(I=I, V=V, f=f, c=c, Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny,
                                         dt=dt, T=T, b=b,
                                         user_action=plot_u, version=version,
                                         safety_factor=safety_factor, C=C)        
    return cpu 


def gaussian(version='vectorized'):
    """
    Initial Gaussian bell in the middle of the domain.
    """
    # Clean up plot files
    for name in glob('tmp_*.png'):
        os.remove(name)

    Lx = 10.
    Ly = 10.
    #c = lambda x, y: cos(x)
    c = 1.

    def I(x, y):
        """Gaussian peak at (Lx/2, Ly/2)."""
        return exp(-0.5*(x-Lx/2.0)**2 - 0.5*(y-Ly/2.0)**2)

    Nx = 30; Ny = 30; T = 30

    
    cpu = visualize(I, 0, 0, c, Lx, Ly, Nx, Ny, 0.1, T, 0., version=version)


def test_constant_solution(version='vectorized', animate=False):
    """Test that u stays constant if I(x,y) = c."""
    Lx = 10.
    Ly = 10.
    c = lambda x, y: cos(x)
    u_exact = 5.
    I = lambda x, y: u_exact
    dt = -1.
    b = 2.

    def printfunc(u, x, xv, y, yv, t, n):
        print u

    Nx = 10; Ny = 10; T = 10

    def assert_no_error(u, x, xv, y, yv, t, n):
        tol = 1E-14
        assert abs((u-u_exact)).max() < tol

    if animate:
        cpu = visualize(I, 0, 0, c, Lx, Ly, Nx, Ny, dt, T, b, version=version) 
    else:
        u, x, xv, y, yv, t, cpu = solver(I, 0, 0, c, Lx, Ly, Nx, Ny, dt, T, b,
                                         user_action=assert_no_error, version=version)



def test_plug(C=1,                   # aximum Courant number
              N=20,                  # spatial resolution
              version='vectorized',  # scheme version
              T=1,                   # end time
              loc='center',          # location of initial condition
              pulse_tp='plugx',      # pulse/init.cond. type
              sigma=0.05,            # width measure of the pulse
              animate=False
              ):
    """
    Check that plug wave splits in two and meet back at the same place.
    The loc parameter can be 'center' or 'left',
    depending on where the initial pulse is to be located.
    The sigma parameter governs the width of the pulse.
    """
    # Use scaled parameters: L=1 for domain length, c_0=1
    # for wave velocity outside the domain.
    L = 10.0
    c = 0.5

    if loc == 'center':
        xc = L/2
    elif loc == 'left':
        xc = 0

    if pulse_tp == 'plugx':
        def I(x, y):
            return 0 if abs(x-xc) > sigma else 1

    elif pulse_tp == 'plugy':
        def I(x, y):
            return 0 if abs(y-xc) > sigma else 1
    else:
        raise ValueError('Wrong pulse_tp="%s"' % pulse_tp)

    dt = -1		# unit Courant number
    
    if animate: 
        cpu = visualize(I=I, V=None, f=None, c=c, Lx=L, Ly=L, Nx=N, Ny=N,
                        dt=dt, T=T, b=0.,
                        version=version,
                        safety_factor=1., C=1.)
    else:
        u, x, xv, y, yv, t, cpu =  solver(I=I, V=None, f=None, c=c, Lx=L, Ly=L, Nx=N, Ny=N,
                                          dt=dt, T=2.0, b=0.,
                                          user_action=None, version=version,
                                          safety_factor=1., C=1.)
        u_exact = array([[I(x_, y_) for y_ in y] for x_ in x])
        tol = 1E-14
        diff = abs(u_exact - u).max()
        assert diff < tol


def standing_undamped_waves(version='vectorized', animate=False):

    mx = 3.0
    my = 3.0
    Lx = 10.0; Ly = 10.0
    kx = (mx*pi)/Lx
    ky = (my*pi)/Ly
    A = 1.0
    c = 2.0
    w = c*sqrt(kx**2 + ky**2)
    u_exact = lambda x, y, t: A*cos(kx*x)*cos(ky*y)*cos(w*t)
    I = lambda x, y: A*cos(kx*x)*cos(ky*y)

    T = 1.0

    def error(u, x, xv, y, yv, t, n):
        u_exact_n = u_exact(xv, yv, t[n])
        # store error squared at time tn for the whole mesh
        e = sum((u_exact_n - u)**2)
        outfile.write('%f' % e + '\n')

    if animate:
        dt = 0.1
        Nx = 20.0; Ny = 20.0
        cpu = visualize(I=I, V=None, f=None, c=c, Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny,
                        dt=dt, T=T, b=0.,
                        version=version,
                        safety_factor=1., C=1., u_exact=u_exact)     

    else:
        h_values = [0.1*2**(-i) for i in range(6)]
        E_values = []
        for h in h_values:
            dt = h/2.
            Nx = 1.0/h; Ny = 1.0/h
            u, x, xv, y, yv, t, cpu = solver(I=I, V=None, f=None, c=c, Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny,
                                                 dt=dt, T=T, b=0.,
                                                 user_action=None, version=version)
            dx = x[1]; dy = y[1]
            u_e = u_exact(xv, yv, t[-1])
            e = sum((u_e - u)**2)
            E = sqrt(dx*dy*dt*e)
            E_values.append(E)

        print E_values
        r = compute_rates(h_values, E_values)
        print 'r: %s' % r
            
            
def compute_rates(h_values, E_values):
    """compute convergence rates"""
    m = len(h_values)
    r = [log(E_values[i-1]/E_values[i])/
         log(h_values[i-1]/h_values[i])
         for i in range(0, m, 1)]
    # Round to two decimals
    r = [round(r_, 2) for r_ in r]
    return r


def manufactured_solution(version='vectorized', animate=False):
    mx = 3.0
    my = 3.0
    Lx = 10.0; Ly = 10.0
    kx = (mx*pi)/Lx
    ky = (my*pi)/Ly
    A = 1.0; B = 1.0
    c = 1.0
    w = c*sqrt(kx**2 + ky**2 - 1)
    b = 2.0

    T = 5.0

    import sympy as sym

    u_exact = lambda x, y, t: (A*cos(w*t) + B*sin(w*t))*exp(-c*t)*cos(kx*x)*cos(ky*y)
    I = lambda x, y: A*cos(kx*x)*cos(ky*y)

    def source_term(bv, kxv, kyv, Av, Bv, wv):
        x, y, t, q, b, kx, ky, A, B, w, fsym, Vsym = sym.symbols('x y t q b kx ky A B w fsym Vsym')
        q = 1.0
        u = (A*sym.cos(w*t)+B*sym.sin(w*t))*sym.exp(-sym.sqrt(q)*t)*sym.cos(kx*x)*sym.cos(ky*y)
        fsym = u.diff(t, 2) + b*u.diff(t, 1) - (q*u.diff(x)).diff(x) - (q*u.diff(y)).diff(y)
        fsym = sym.simplify(fsym)
        Vsym = u.diff(t)
        f = sym.lambdify((x, y, t), fsym.subs([(b,bv),(kx,kxv),(ky,kyv),(A,Av),(B,Bv),(w,wv)]), 'numpy')
        V = sym.lambdify((x, y), Vsym.subs([(b,bv),(kx,kxv),(ky,kyv),(A,Av),(B,Bv),(w,wv),(t,0)]), 'numpy')
        return f, V

    f, V = source_term(b, kx, ky, A, B, w)

    if animate:
        Nx = 10; Ny = 10
        dt = 0.1
        cpu = visualize(I=I, V=V, f=f, c=c, Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny,
                        dt=dt, T=T, b=b,
                        version=version,
                        safety_factor=1., C=1., u_exact=u_exact)   

    else:
        h_values = [0.1*2**(-i) for i in range(6)]
        E_values = []
        for h in h_values:
            dt = h
            Nx = 1.0/h; Ny = 1.0/h
            u, x, xv, y, yv, t, cpu = solver(I=I, V=V, f=f, c=c, Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny,
                                                 dt=dt, T=T, b=0.,
                                                 user_action=None, version=version)
            dx = x[1]; dy = y[1]
            u_e = u_exact(xv, yv, t[-1])
            e = sum((u_e - u)**2)
            E = sqrt(dx*dy*dt*e)
            E_values.append(E)

        print E_values
        r = compute_rates(h_values, E_values)
        print 'r: %s' % r


def physical_problem(version='vectorized', bottom='Gaussian'):

    T = 5.0   
    Lx = 5.0; Ly = 5.0
    g = 9.81
    b = 0.0

    f = 0
    V = 0

    Nx = 20.0; Ny = 20.0
    dt = -0.9

    if bottom == 'Gaussian':
        I_0 = 1.2; I_a = 0.5 ; I_m = 0; I_s = 0.1
        B_0 = 0.0; B_a = 0.7; B_mx = Lx/2.0 ; B_my = Ly/2.0 ; B_s = 0.3 ; b_scale = 1
        I = lambda x, y: I_0 + I_a*exp(-((x-I_m)/I_s)**2)
        B = lambda x, y: B_0 + B_a*exp(-((x-B_mx)/B_s)**2 - ((y-B_my)/(b_scale*B_s))**2)
    
    elif bottom == 'cosine_hat':
        I_0 = 1.3; I_a = 0.5 ; I_m = Lx ; I_s = 0.5
        B_0 = 0; B_a = 1.1; B_mx = Lx/2.0  ; B_my = Lx/2.0 ; B_s = 0.5
        I = lambda x, y: I_0 + I_a*exp(-((x-I_m)/I_s)**2)
        B = vectorize(lambda x, y: B_0 + B_a*cos(pi*(x-B_mx)/(2*B_s))*cos(pi*(y-B_my)/(2*B_s)) \
                         if 0 <= sqrt((x-Lx/2.0)**2+(y-Ly/2.0)**2) <= B_s else B_0)

    elif bottom == 'box':
        I_0 = 1.3; I_a = 0.5 ; I_m = Lx ; I_s = 0.5
        B_0 = 0; B_a = 1.4; B_mx =Lx/2.0  ; B_my = Lx/2.0 ; B_s = 0.3; b_scale = 1.
        I = lambda x, y: I_0 + I_a*exp(-((x-I_m)/I_s)**2)
        B = vectorize(lambda x, y: B_0 + B_a \
                         if B_mx - B_s <= x <= B_mx + B_s and B_my-b_scale*B_s <=y<= B_my+b_scale*B_s else B_0)

    else:
        raise ValueError('Wrong pulse_tp="%s"' % pulse_tp)

    c = lambda x,y:  sqrt(9.81*(I(x,y)-B(x,y)))   

    cpu = visualize(I=I, V=V, f=f, c=c, Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny,
                    dt=dt, T=T, b=b,
                    version=version,
                    safety_factor=1., C=1., u_exact=None, B=B)    


    

if __name__ == '__main__':
    pass
    # gaussian - not part of project
    #gaussian(version='vectorized')
    
    # test constant solution: py.test 2dwave_project.py for pytest
    #test_constant_solution(version='scalar', animate=True)

    # test plug-wave: 'plugx' for x-direction, 'plugy' for y-direction
    #test_plug(version='scalar', animate=False, pulse_tp='plugx')
    
    # animate=False to compute convergence rates
    #standing_undamped_waves(version='vectorized', animate=False)

    # animate=False to compute convergence rates
    #manufactured_solution(version='vectorized', animate=False)

    physical_problem(bottom='Gaussian')
    
    #movie('waveplot_*.png', encoder='convert', fps=25, output_file='animation.gif')	# Make a gif


    

