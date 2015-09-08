from numpy import *
import matplotlib.pyplot as plt

def simulate(
    beta=0.9,                 # dimensionless parameter
    Theta=30,                 # initial angle in degrees
    epsilon=0,                # initial stretch of wire
    num_periods=6,            # simulate for num_periods
    time_steps_per_period=60, # time step resolution
    plot=True,                # make plots or not.
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
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Motion of elastic pendulum')
        plt.show()
          
        plt.plot(t, theta)
        plt.xlabel('t')
        plt.ylabel('Degrees')
        plt.title('Angular motion of elastic pendulum')
        plt.show()
        
        if rad2deg(Theta) < 10:
            non_elastic_pendulum = rad2deg(Theta)*cos(t)
            plt.plot(t, non_elastic_pendulum, 'b-', t, theta, 'g-')
            plt.legend(['Non-elastic pendulum', 'Elastic pendulum, theta0 < 10'])
            plt.title('Comparing elastic pendulum for small initial inclinations with non-elastic pendulum')
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
    for differendt values of beta and epsilon
    """
    print '=== Comparing exact and numerical solution for pure vertical motion ==='
    beta_eps =  [0.1, 0.3, 0.5, 0.7, 0.9]
    for beta in beta_eps:
        for eps in beta_eps:
            t, x, y, theta = simulate(beta, Theta=0, epsilon=eps, plot=False, 
                             time_steps_per_period=300)
            w = sqrt(beta/(1-beta))
            exact = y[0]*cos(w*t)
            E = max(abs(y-exact))	# Error
            print 'Beta = %.1f, Epsilon = %.1f: Error: %e' % (beta, eps, E)


def demo(beta, Theta):
    """
    Demo of numerical solution to elastic pendulum
    """
    t, x, y, theta = simulate(Theta=Theta, beta=beta, 
                              time_steps_per_period=600, 
                              num_periods=3)



# === main ===
#test_func1()
#test_func2()
demo(0.9, 20)



"""
Output when run as
elastic_penulum.py 

- from test_func1:

=== Testing exact solution: u(t) = t**2 + t + 1 ===
Initial conditions u(0) = 1, u'(0) = 1:
residual step1: 0
residual: 0

- from test_func2:

=== Comparing exact and numerical solution for pure vertical motion ===
Beta = 0.1, Epsilon = 0.1: Error: 2.242137e-06
Beta = 0.1, Epsilon = 0.3: Error: 6.726412e-06
Beta = 0.1, Epsilon = 0.5: Error: 1.121069e-05
Beta = 0.1, Epsilon = 0.7: Error: 1.569496e-05
Beta = 0.1, Epsilon = 0.9: Error: 2.017924e-05
Beta = 0.3, Epsilon = 0.1: Error: 1.847267e-05
Beta = 0.3, Epsilon = 0.3: Error: 5.541802e-05
Beta = 0.3, Epsilon = 0.5: Error: 9.236336e-05
Beta = 0.3, Epsilon = 0.7: Error: 1.293087e-04
Beta = 0.3, Epsilon = 0.9: Error: 1.662541e-04
Beta = 0.5, Epsilon = 0.1: Error: 6.605847e-05
Beta = 0.5, Epsilon = 0.3: Error: 1.981754e-04
Beta = 0.5, Epsilon = 0.5: Error: 3.302924e-04
Beta = 0.5, Epsilon = 0.7: Error: 4.624093e-04
Beta = 0.5, Epsilon = 0.9: Error: 5.945263e-04
Beta = 0.7, Epsilon = 0.1: Error: 2.345218e-04
Beta = 0.7, Epsilon = 0.3: Error: 7.035653e-04
Beta = 0.7, Epsilon = 0.5: Error: 1.172609e-03
Beta = 0.7, Epsilon = 0.7: Error: 1.641652e-03
Beta = 0.7, Epsilon = 0.9: Error: 2.110696e-03
Beta = 0.9, Epsilon = 0.1: Error: 1.835250e-03
Beta = 0.9, Epsilon = 0.3: Error: 5.505751e-03
Beta = 0.9, Epsilon = 0.5: Error: 9.176251e-03
Beta = 0.9, Epsilon = 0.7: Error: 1.284675e-02
Beta = 0.9, Epsilon = 0.9: Error: 1.651725e-02
"""





