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
from scitools.std import *

def solver(I, V, f, c, Lx, Ly, Nx, Ny, dt, T, b,
           user_action=None, version='scalar',
           safety_factor=1.0, C=1.0):

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

    stability_limit = (dx*safety_factor*C)/c_max
    #stability_limit = (dx*safety_factor)/c_max
    
    if dt <= 0:                # max time step?
        safety_factor = -dt    # use negative dt as safety factor
        dt = safety_factor*stability_limit
    elif dt > stability_limit:
        print 'error: dt=%g exceeds the stability limit %g' % \
              (dt, stability_limit)
    Nt = int(round(T/float(dt)))
    t = linspace(0, Nt*dt, Nt+1)    # mesh points in time

    # Treat c(x) as array
    if isinstance(c, (float,int)):
        c = np.zeros((Nx+1,Ny+1)) + c
    elif callable(c):
        # Call c(x) and fill array c
        c_ = np.zeros((Nx+1,Ny+1))
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
    return u, x, y, t, t1 - t0



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


def gaussian(version='vectorized'):
    """
    Initial Gaussian bell in the middle of the domain.
    plot_method=1 applies mesh function, =2 means surf, =0 means no plot.
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


def test_constant_solution(version='scalar', animate=False):
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
        print cpu   
    else:
        u, dt, cpu = solver(I, 0, 0, c, Lx, Ly, Nx, Ny, dt, T, b,
                            user_action=assert_no_error, version=version)



def pulse(C=1,            # aximum Courant number
          N=20,         # spatial resolution
          version='vectorized',
          T=1,            # end time
          loc='center',     # location of initial condition
          pulse_tp='plugx',  # pulse/init.cond. type
          slowness_factor=2, # wave vel. in right medium
          medium=[0.7, 0.9], # interval for right medium
          skip_frame=1,      # skip frames in animations
          sigma=0.05,        # width measure of the pulse
          animate=False
          ):
    """
    Various peaked-shaped initial conditions on [0,1].
    Wave velocity is decreased by the slowness_factor inside
    medium. The loc parameter can be 'center' or 'left',
    depending on where the initial pulse is to be located.
    The sigma parameter governs the width of the pulse.
    """
    # Use scaled parameters: L=1 for domain length, c_0=1
    # for wave velocity outside the domain.
    L = 1.0
    c_0 = 1.0

    if loc == 'center':
        xc = L/2
    elif loc == 'left':
        xc = 0

    if pulse_tp in ('gaussian','Gaussian'):
        def I(x):
            return exp(-0.5*((x-xc)/sigma)**2)

    elif pulse_tp == 'plugx':
        def I(x, y):
            return 0 if abs(x-xc) > sigma else 1

    elif pulse_tp == 'plugy':
        def I(x, y):
            return 0 if abs(y-xc) > sigma else 1

    elif pulse_tp == 'cosinehat':
        def I(x):
            # One period of a cosine
            w = 2
            a = w*sigma
            return 0.5*(1 + cos(pi*(x-xc)/a)) \
                   if xc - a <= x <= xc + a else 0

    elif pulse_tp == 'half-cosinehat':
        def I(x):
            # Half a period of a cosine
            w = 4
            a = w*sigma
            return cos(pi*(x-xc)/a) \
                   if xc - 0.5*a <= x <= xc + 0.5*a else 0
    else:
        raise ValueError('Wrong pulse_tp="%s"' % pulse_tp)

    c = c_0 
    """
    def c(x, y):
        return c_0/slowness_factor \
               if medium[0] <= x.any() <= medium[1] else c_0
    """

    def printfunc(u, x, xv, y, yv, t, n):
        print u[0,:]
        print

    dt = 0.05 # choose the stability limit with given Nx, worst case c
    
    

    if animate: 
        cpu = visualize(I=I, V=None, f=None, c=c, Lx=L, Ly=L, Nx=N, Ny=N,
                        dt=dt, T=T, b=0.,
                        version=version,
                        safety_factor=1., C=1.)
    else:
        u, dt, cpu =  solver(I=I, V=None, f=None, c=c, Lx=L, Ly=L, Nx=N, Ny=N,
                             dt=dt, T=T, b=0.,
                             user_action=printfunc, version=version,
                             safety_factor=1., C=1.)


def standing_undamped_waves(version='scalar', animate=False):

    mx = 3.0
    my = 3.0
    Lx = 10.0; Ly = 10.0
    kx = (mx*pi)/Lx
    ky = (my*pi)/Ly
    A = 1.0
    w = pi
    u_exact = lambda x, y, t: A*cos(kx*x)*cos(ky*y)*cos(w*t)
    I = lambda x, y: A*cos(kx*x)*cos(ky*y)

    c = 1.0
    Nx = 20.0; Ny = 20.0
    T = 3.0
    
    #global e_max
    #e_max = 0

    def error(u, x, xv, y, yv, t, n):
        e = abs(u_exact(x, y, t[n]) - u)
        global e_max
        e_max = max(e_max, e.max())  

    if animate:
        dt = 0.1
        cpu = visualize(I=I, V=None, f=None, c=c, Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny,
                        dt=dt, T=T, b=0.,
                        version=version,
                        safety_factor=1., C=1.)     

    else:
        dt_values = [0.1*2**(-i) for i in range(6)]
        E_values = []
        for dt in dt_values:
            u, x, y, t, cpu = solver(I=I, V=None, f=None, c=c, Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny,
                                     dt=dt, T=T, b=0.,
                                     user_action=None, version=version)
            e = sum((u_exact(x, y, t[-1]) - u)**2)
            dx = x[1]-x[0]; dy = y[1]-y[0]; dt = t[1]-t[0]
            e = sqrt(dx*dy*dt*e)
            E_values.append(e)

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
            



"""
def source_term(L):
    
    Adapt source term to exact solution u_exact using sympy
    
    un, xn, yn, tn, fn, w, qn, Ln = sym.symbols('un xn tn fn w qn Ln')
    #qn = 1 + (xn-(Ln/2))**4		# a)
    qn = 1 + sym.cos(sym.pi*xn/Ln)	# b)
    w = 1
    un = sym.cos(sym.pi*xn/Ln)*sym.cos(w*tn)
    fn = un.diff(tn, 2) - (qn*un.diff(xn)).diff(xn)
    # convert to callable python lambda function
    f = sym.lambdify((xn, yn, tn), sym.simplify(fn.subs(Ln,L)))
    return f
"""


def visualize(I, V, f, c, Lx, Ly, Nx, Ny, dt, T, b, 
              version, safety_factor=1., C=1.,
              u_exact=None):

    from mpl_toolkits.mplot3d import axes3d
    import matplotlib.pyplot as plt
    import numpy as np
    import time

    plt.ion()

    if u_exact:
        fig = plt.figure(figsize=plt.figaspect(0.5))
    else:
        fig = plt.figure()

    ax = axes3d.Axes3D(fig)

    global wframe
    wframe = None

    def plot_u(u, x, xv, y, yv, t, n):
        X, Y = meshgrid(x,y)       
      
        #ax.set_zlim3d(-10, 10)
        global wframe
        oldcol = wframe   
        
        wframe = ax.plot_wireframe(X, Y, u)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('u')


        # Remove old line collection before drawing
        if oldcol is not None:
            ax.collections.remove(oldcol)

        plt.draw()
        time.sleep(0.02)
   
    # plot u and u_exact 
    """
    def plot_u_exact(u, x, xv, y, yv, t, n):
    """
              

    u, dt, cpu = solver(I=I, V=V, f=f, c=c, Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny,
                        dt=dt, T=T, b=b,
                        user_action=plot_u, version=version,
                        safety_factor=safety_factor, C=C)
    return cpu 


    

if __name__ == '__main__':
    #gaussian(version='vectorized')
    #test_constant_solution(version='scalar')
    #pulse(version='scalar', animate=True, T=5., pulse_tp='plugy')
    standing_undamped_waves(version='vectorized', animate=False)
