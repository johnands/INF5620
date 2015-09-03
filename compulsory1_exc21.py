from numpy import *
import matplotlib.pyplot as plt

def simulate(
    beta=0.9,                 # dimensionless parameter
    Theta=30,                 # initial angle in degrees
    epsilon=0,                # initial stretch of wire
    num_periods=6,            # simulate for num_periods
    time_steps_per_period=60, # time step resolution
    plot=True,                # make plots or not
    ):

    """
    Solve ODE u'' + w**2u = 0 for elastic pendulum.
    Return arrays, for t, x, y and theta.
    If plot=True, plot y versus x, and theta (in degrees) vs t.
    """

    # numerical initialization
    P  = 2*pi				# period of oscillations for classical, non-elastic pendulum
    T  = num_periods*P			# total simulation time
    dt = float(P/time_steps_per_period)	# time step
    N  = int(round(T/dt))		# number of mesh points
    x  = zeros(N+1)
    y  = zeros(N+1)
    theta  = zeros(N+1)
    t  = linspace(0, T, N+1)

    Theta = deg2rad(Theta)

    # initial conditions
    x[0] = (1+epsilon)*sin(Theta)
    y[0] = 1 - (1+epsilon)*cos(Theta)
    print x[0]
    print y[0]

    # scaled rope length L(t)
    L = sqrt(x[0]**2 + (y[0]-1)**2)

    # first time step
    x[1] = x[0] - 0.5*dt**2*(beta/(1-beta))*(1-(beta/L))*x[0]
    y[1] = y[0] - 0.5*dt**2*(beta/(1-beta))*(1-(beta/L))*(y[0]-1) - 0.5*dt**2*beta
			
    # integration loop
    for i in range(1, N):
        L = sqrt(x[i]**2 + (y[i]-1)**2)
        x[i+1] = 2*x[i] - x[i-1] - dt**2*(beta/(1-beta))*(1-(beta/L))*x[i]
        y[i+1] = 2*y[i] - y[i-1] - dt**2*(beta/(1-beta))*(1-(beta/L))*(y[i]-1) - dt**2*beta
        

    theta = arctan(x/(1-y))

    # plot
    if plot == True:
        plt.plot(x, y)
        plt.show()
        
        plt.plot(t,y)
        plt.show()
      
        
        plt.plot(t, theta)
        plt.show()

    return t, x, y, theta


t, x, y, theta = simulate()


