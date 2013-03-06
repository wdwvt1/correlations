#!/usr/bin/env/python
#file created 3/4/2013

__author__ = "Will Van Treuren"
__copyright__ = "Copyright 2013, Will Van Treuren"
__credits__ = ["Will Van Treuren"]
__license__ = "GPL"
__url__ = ''
__version__ = ".9-Dev"
__maintainer__ = "Will Van Treuren"
__email__ = "wdwvt1@gmail.com"

"""
Code for generating OTU tables from Lokta-Volterra models.
This was inital playing around trying to understand what was going on. Saved
in case something of value is hidden here...
"""

################################################################################
# Predator-prey relationship (2 species)
# Adapted from http://www.scipy.org/Cookbook/LoktaVolterraTutorial
################################################################################


'''
du/dt =  a*u -   b*u*v
dv/dt = -c*v + d*b*u*v 

u: prey population
v: predator population
a: constant, natural growth rate of prey population
b: constant, death rate of prey due to predation
c: constant, death rate of predators 
d: constant, number of prey required to support a predator

We will use X=[u, v] to describe the state of both populations.

Two fixed points (dX_dt=[0,0]):
 [u,v] = [0,0]
 [u,v] = [c/(d*b),a/b]

Near the fixed points the system can be linearized:
 dX_dt = A_f*X where A_f is the Jacobian evaluated at f. 
 assuming
 a = 1.
 b = 0.1
 c = 1.5
 d = 0.75
 Then
 A_f0 = d2X_dt2(X_f0) = [[1,0],[0,-1.5]]
 A_f1 = d2X_dt2(X_f1) = [[0,-2],[.75,0]]

Eigenvalues of A_fi:
 lambda1, lambda2 = linalg.eigvals(A_f1) = (1.22474j, -1.22474j)

Because the eigenvals are imaginary we have periodic behavior. Period:
 T_f1 = 2*pi/abs(lambda1) = 5.130199

Using scipy.integrate to solve the ODES. 

'''



from scipy import integrate
from numpy import array, arange, linspace, where, eye
import matplotlib.pyplot as plt



# def lv_pred_prey(u_t0, v_t0, a, b, c, d, lb=0, ub=15, ts=1000):
#     '''Simuate predator prey relationship via Lokta-Volterra model.
#     Inputs:
#      u_t0 - int, initial prey population
#      v_t0 - int, initial predator population
#      a - float, constant natural growth rate of prey population
#      b - float, constant death rate of prey due to predation
#      c - float, constant death rate of predators 
#      d - float, constant number of prey required to support a predator
#      ub, lb, ts - int, bounds and timestep size.
#     '''
#     # have to define this function internally because odeint requires the
#     # dX_dt function to accept only X and t, but would otherwise have to define
#     # constants a-d. tried odint(args=(a,b,c,d)) but it passes them in a way
#     # that i don't understand so this method will have to suffice for now.
#     def _dX_dt(X, t=0):
#         """Growth rate of u,v. X=[u,v]."""
#         # t parameter is utilized by the scipy.integrate.odeint function
#         return array([a*X[0] - b*X[0]*X[1],-c*X[1] + d*b*X[0]*X[1]])
    
#     t = linspace(lb, ub,  ts) #timesteps to eval lv at
#     X0 = array([u_t0, v_t0]) #initial conditions
#     X = integrate.odeint(_dX_dt, X0, t)
#     return X.T # X is tsX2 and we want 2Xts

# def lv_multispecies(u_t0, v_t0, r_t0, a,b,c,d,e,f,lb=0, ub=65, ts=1000):
#     '''Simulate species according to n-dimensional Lokta-Volterra model.
#     '''

#     def _dX_dt(X,t=0):
#         '''
#         a - float, constant natural growth rate of prey population
#         b - float, constant death rate of prey due to predation
#         c - float, constant death rate of predators 
#         d - float, constant number of prey required to support a predator
#         e - super-pred
#         du/dt =  a*u -   b*u*v
#         dv/dt = -c*v + d*b*u*v 
#         dr/dt = -e*r + f*u*v
#         dy(1) = y(1)*(2 - 2*y(2))
#         dy(2) = y(2)*(1 + 2*y(1) - y(2) - 2*y(3))
#         dy(3) = y(3)*(4 + 2*y(2) - 2*y(4) - 2*y(5) - 2*y(6))
#         '''
#         return array([a*X[0] - b*X[0]*X[1] - .01*X[2],-c*X[1] + d*b*X[0]*X[1],
#             -e*X[2]+f*X[0]*X[1]])

#     t = linspace(lb, ub,  ts) #timesteps to eval lv at
#     X0 = array([u_t0, v_t0, r_t0]) #initial conditions
#     X = integrate.odeint(_dX_dt, X0, t)
#     return X.T # X is tsX2 and we want 2Xts

def dX_dt_template(C):
    '''Create a function that scipy.integrate.odeint can use for LV models.
    The interaction matrix specifies the coefficients for each term in the 
    derivative of the given species population. All equations take the form of:
        dxi/dt = xi * (a*x0 + b*x1 + c*x2 +...i +...n*xn) 
    coefs = 0 imply no direct interaction between species k and species i.
    Inputs:
     interaction_matrix - nXn array.
    '''
    # Assume the following notation:
    # X = [x0, x1 ... xn] vector of populations for each species
    # C = [[a0,b0,c0,...n0],
    #      [a1,b1,c1,...n0],
    #              .
    #              .
    #              .
    #      [an,bn,cn,...nn]]
    # specify rows of C by Ci, entries by Cij
    # Then:
    # xi*(X*Ci - xi*Cii + Cii) = xi*(ai*x0 + bi*x1 + ... ni*xn - ii*xi + ii) =
    # xi*(ai*x0 + ... ii ... + ni*xn) = dxi/dt
    # dxi_dt = X[i]*(C[i]*X - C[i][i]*X[i] + C[i][i])
    # we create the following matrix thats multiplied element wise rather than 
    # with matrix multiplication. 
    # x0     [c00,c01*x1,c02*x2,...,c0n*xn],
    #  .     [c10*x0,c11,c12*x2,...,c1n*xn],
    #  .             .
    #  .  *          .
    #  .             .
    # xn     [cn0*x0,cn1*x1,cn2*x2,...,cnn]]
    # sum along cols axis. X must be a 2D array because of the vagueries of 
    # numpy 
    return lambda X, t=0: (array([X]).T*where(eye(C.shape[0]), C, C*X)).sum(1)

def lokta_volterra(dX_dt, X0, lb, ub, ts):
    '''Simulate species interactions via Lokta-Volterra model.
    Inputs:
     dX_dt - function, must take a vector X (species abundances) and a scalar t
     for timestep. t must default to 0.
     X0 - 1d arr, initial populations. 
     ub, lb, ts - int, bounds and timestep size.
    '''
    t = linspace(lb, ub,  ts) #timesteps to eval lv at
    X = integrate.odeint(dX_dt, X0, t)
    return X.T #transpose to make otusXsamples


# du/dt =  a*u -   b*u*v
# dv/dt = -c*v + d*b*u*v 


#  a = 1.
#  b = 0.1
#  c = 1.5
#  d = 0.75

C = array([[1., -.1],
           [.075, -1.5]])

f = dX_dt_template(C)

def dX_dt(X, t=0):
    """Growth rate of u,v. X=[u,v]."""
    # t parameter is utilized by the scipy.integrate.odeint function
    return array([C[0][0]*X[0] + C[0][1]*X[0]*X[1],C[1][1]*X[1] + .075*X[0]*X[1]])

#X = lokta_volterra(dX_dt, array([10,5]), 0, 15, 1000)
Y = lokta_volterra(f, array([10,5]), 0, 15, 1000)




t = linspace(0,15,1000)
rabbits, foxes = X
f1 = plt.figure()
plt.plot(t, rabbits, 'r-', label='Rabbits')
plt.plot(t, foxes  , 'b-', label='Foxes')
plt.grid()
plt.legend(loc='best')
plt.xlabel('time')
plt.ylabel('population')
plt.title('Evolution of fox and rabbit populations')
f1.savefig('rabbits_and_foxes_1.png')
plt.show()

rabbits, foxes = Y
f1 = plt.figure()
plt.plot(t, rabbits, 'r-', label='Rabbits')
plt.plot(t, foxes  , 'b-', label='Foxes')
plt.grid()
plt.legend(loc='best')
plt.xlabel('time')
plt.ylabel('population')
plt.title('Evolution of fox and rabbit populations')
f1.savefig('rabbits_and_foxes_1.png')
plt.show()

# this will cause rabbit population to dip below 0 by time 65
# X = lv_multispecies(10,5,5,1,.1,1.5,.75,1.0,.20)
# t = linspace(0,65,1000)
# rabbits, foxes, space_crabs = X
# f1 = plt.figure()
# plt.plot(t, rabbits, 'r-', label='Rabbits')
# plt.plot(t, foxes  , 'b-', label='Foxes')
# plt.plot(t, space_crabs  , 'g-', label='Giant space crab')

# plt.grid()
# plt.legend(loc='best')
# plt.xlabel('time')
# plt.ylabel('population')
# plt.title('Evolution of fox and rabbit populations')
# f1.savefig('rabbits_and_foxes_crab_1.png')
# plt.show()

# # to get the example provided
# '''
# X = lv_pred_prey(10,5,1,.1,1.5,.75)
# t = linspace(0,15,1000)
# rabbits, foxes = X
# f1 = plt.figure()
# plt.plot(t, rabbits, 'r-', label='Rabbits')
# plt.plot(t, foxes  , 'b-', label='Foxes')
# plt.grid()
# plt.legend(loc='best')
# plt.xlabel('time')
# plt.ylabel('population')
# plt.title('Evolution of fox and rabbit populations')
# f1.savefig('rabbits_and_foxes_1.png')
#'''
#




# def dX_dt(X, t=0):
#     """Growth rate of u,v. X=[u,v]."""
#     # t parameter is utilized by the scipy.integrate.odeint function
#     return array([a*X[0] - b*X[0]*X[1],-c*X[1] + d*b*X[0]*X[1]])


# def d2X_dt2(X, t=0):
#     """Return the Jacobian matrix evaluated in X. """
#     # again with a t parameter that seems unutilized
#     return array([[a - b*X[1], -b*X[0]], [b*d*X[1], -c + b*d*X[0]]])  


# The populations are indeed periodic, and their period is near to the T_f1 we calculated.
# 
# == Plotting direction fields and trajectories in the phase plane ==
# 
# We will plot some trajectories in a phase plane for different starting
# points between X__f0 and X_f1.
# 
# We will use matplotlib's colormap to define colors for the trajectories.
# These colormaps are very useful to make nice plots.
# Have a look at [http://www.scipy.org/Cookbook/Matplotlib/Show_colormaps ShowColormaps] if you want more information.
# 
# values  = linspace(0.3, 0.9, 5)                          # position of X0 between X_f0 and X_f1
# vcolors = plt.cm.autumn_r(linspace(0.3, 1., len(values)))  # colors for each trajectory

# f2 = plt.figure()

# #-------------------------------------------------------
# # plot trajectories
# for v, col in zip(values, vcolors): 
#     X0 = v * X_f1                               # starting point
#     X = integrate.odeint( dX_dt, X0, t)         # we don't need infodict here
#     plt.plot( X[:,0], X[:,1], lw=3.5*v, color=col, label='X0=(%.f, %.f)' % ( X0[0], X0[1]) )

# #-------------------------------------------------------
# # define a grid and compute direction at each point
# ymax = plt.ylim(ymin=0)[1]                        # get axis limits
# xmax = plt.xlim(xmin=0)[1] 
# nb_points   = 20                      

# x = linspace(0, xmax, nb_points)
# y = linspace(0, ymax, nb_points)

# X1 , Y1  = meshgrid(x, y)                       # create a grid
# DX1, DY1 = dX_dt([X1, Y1])                      # compute growth rate on the gridt
# M = (hypot(DX1, DY1))                           # Norm of the growth rate 
# M[ M == 0] = 1.                                 # Avoid zero division errors 
# DX1 /= M                                        # Normalize each arrows
# DY1 /= M                                  

# #-------------------------------------------------------
# # Drow direction fields, using matplotlib 's quiver function
# # I choose to plot normalized arrows and to use colors to give information on
# # the growth speed
# plt.title('Trajectories and direction fields')
# Q = plt.quiver(X1, Y1, DX1, DY1, M, pivot='mid', cmap=plt.cm.jet)
# plt.xlabel('Number of rabbits')
# plt.ylabel('Number of foxes')
# plt.legend()
# plt.grid()
# plt.xlim(0, xmax)
# plt.ylim(0, ymax)
# f2.savefig('rabbits_and_foxes_2.png')
# # 
# # 
# # We can see on this graph that an intervention on fox or rabbit populations can
# # have non intuitive effects. If, in order to decrease the number of rabbits,
# # we introduce foxes, this can lead to an increase of rabbits in the long run,
# # if that intervention happens at a bad moment.
# # 
# # 
# # == Plotting contours ==
# # 
# # We can verify that the function IF defined below remains constant along a trajectory:
# # 
# def IF(X):
#     u, v = X
#     return u**(c/a) * v * exp( -(b/a)*(d*u+v) )

# # We will verify that IF remains constant for different trajectories
# for v in values: 
#     X0 = v * X_f1                               # starting point
#     X = integrate.odeint( dX_dt, X0, t)         
#     I = IF(X.T)                                 # compute IF along the trajectory
#     I_mean = I.mean()
#     delta = 100 * (I.max()-I.min())/I_mean
#     print 'X0=(%2.f,%2.f) => I ~ %.1f |delta = %.3G %%' % (X0[0], X0[1], I_mean, delta)

# # >>> X0=( 6, 3) => I ~ 20.8 |delta = 6.19E-05 %
# #     X0=( 9, 4) => I ~ 39.4 |delta = 2.67E-05 %
# #     X0=(12, 6) => I ~ 55.7 |delta = 1.82E-05 %
# #     X0=(15, 8) => I ~ 66.8 |delta = 1.12E-05 %
# #     X0=(18, 9) => I ~ 72.4 |delta = 4.68E-06 %
# # 
# # Potting iso-contours of IF can be a good representation of trajectories,
# # without having to integrate the ODE
# # 
# #-------------------------------------------------------
# # plot iso contours
# nb_points = 80                              # grid size 

# x = linspace(0, xmax, nb_points)    
# y = linspace(0, ymax, nb_points)

# X2 , Y2  = meshgrid(x, y)                   # create the grid
# Z2 = IF([X2, Y2])                           # compute IF on each point

# f3 = plt.figure()
# CS = plt.contourf(X2, Y2, Z2, cmap=plt.cm.Purples_r, alpha=0.5)
# CS2 = plt.contour(X2, Y2, Z2, colors='black', linewidths=2. )
# plt.clabel(CS2, inline=1, fontsize=16, fmt='%.f')
# plt.grid()
# plt.xlabel('Number of rabbits')
# plt.ylabel('Number of foxes')
# plt.ylim(1, ymax)
# plt.xlim(1, xmax)
# plt.title('IF contours')
# f3.savefig('rabbits_and_foxes_3.png')
# plt.show()
# # 
# # 
# # # vim: set et sts=4 sw=4: