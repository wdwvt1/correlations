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

'''Code for generating n-dimensional Lokta-Volterra relationships. 
This code was guided heavily by the scipy tutorial available at
http://www.scipy.org/Cookbook/LoktaVolterraTutorial. 

To generate relationships an interaction matrix (denoted C) must be generated. 
The interaction matrix is n*n and specifies the coefficients for each equation. 
The traditional LV model takes the following two species form:
    
    du/dt =  a*u -   b*u*v
    dv/dt =  d*b*u*v - c*v

    u - prey population
    v - predator population
    a - constant, natural growth rate of prey population
    b - constant, death rate of prey due to predation
    c - constant, death rate of predators 
    d - constant, number of prey required to support a predator

In the n-dimensional model we have equations of the form:
    
    dxi/dt = xi * (a*x0 + b*x1 + c*x2 +...i +...n*xn) 

    xi - species xi
    a,b,c... - coefficients determining interaction between xi and other species

Notice that the n-dimensional equation has n-1 terms of the form j*xi*xj and 
one term i*xi where xi, xj are species and i,j are coefficients of interaction
between them. 

To recreate the scipy tutorial data you can use the following:

C = array([[1., -.1],
           [.075, -1.5]])
f = dX_dt_template(C)
Y = lokta_volterra(f, array([10,5]), 0, 15, 1000)
'''

from numpy import array, where, eye, linspace
from scipy import integrate

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
    # sum along cols axis. X must be a 2D array because of the vagaries of 
    # numpy elementwise multiplication and array shape.
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

