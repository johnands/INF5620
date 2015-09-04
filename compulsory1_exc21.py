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
        
    # inclination as function of time in degrees
    theta = rad2deg(arctan(x/(1-y)))

    # plots
    if plot == True:
        plt.gca().set_aspect('equal')
        plt.plot(x, y)
        plt.show()
          
        plt.plot(t, theta)
        plt.show()
        
        if rad2deg(Theta) < 10:
            non_elastic_pendulum = rad2deg(Theta)*cos(t)
            plt.plot(t, non_elastic_pendulum, 'b-', t, theta, 'g-')
            plt.legend(['Non-elastic pendulum', 'Elastic pendulum, theta0 < 10'])
            plt.show()
        

    return t, x, y, theta


def test_func1():
    """
    Test whether zero initial inclination and stretch leads to 
    pendulum staying at origin
    """
    t, x, y, theta = simulate(Theta=0, epsilon=0, plot=False)
    print "=== Testing whether x(t) and y(t) are zero at all times ==="
    print 'Maximum value of x(t): %e' % abs(max(x))
    print 'Maximum value of y(t): %e' % abs(max(y))

def test_func2():
    """
    Test for pure vertical motion of elastic pendulum
    """
    t, x, y, theta = simulate(Theta=0, epsilon=0.1, plot=False, 
                              time_steps_per_period=200)
    beta = 0.9
    freq = sqrt(beta/(1-beta))
    exact = y[0]*cos(freq*t)
    plt.plot(x, y)
    #plt.show()
    plt.plot(x, exact)
    #plt.show()
    E = abs(y-exact)
    plt.plot(E)
    plt.show()

def demo(beta, Theta):
    t, x, y, theta = simulate(Theta=Theta, beta=beta, 
                              time_steps_per_period=600, 
                              num_periods=3)



# main
#t, x, y, theta = simulate(Theta = 8)
#test_func1()
#test_func2()
demo(0.9, 15)









