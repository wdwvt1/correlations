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
    
    dxi/dt = xi * (alpha_i + ci1*x1 + ci2*x2 + ci3*x3 + ... ciixi + ... cin*xn) 

    xi - species xi
    alpha_i is the coefficient which provides the growth rate of species xi 
    in the absense of interactions. basic first order. 
    cij terms are teh coefficients of interaction between species xi and species
    xj. cii is the coefficient of interaction of species xi with itself. this 
    creates second order effects. 

Notice that the n-dimensional equation has n- terms of the form j*xi*xj and 
one term i*xi where xi, xj are species and i,j are coefficients of interaction
between them. 

To recreate the scipy tutorial data you can use the following:

C = array([[1., 0, -.1],
           [-1.5, 0.075, 0]])
f = dX_dt_template(C)
Y = lokta_volterra(f, array([10,5]), 0, 15, 1000)
'''

from numpy import array, where, eye, linspace, matrix, hstack
from scipy import integrate

def dX_dt_template(C):
    '''Create a function that scipy.integrate.odeint can use for LV models.
    The interaction matrix specifies the coefficients for each term in the 
    derivative of the given species population. All equations take the form of:
    dxi/dt = xi * (alpha_i + ci1*x1 + ci2*x2 + ci3*x3 + ... ciixi + ... cin*xn) 
    coefs = 0 imply no direct interaction between species k and species i.
    Inputs:
     interaction_matrix - nXn+1 array.
    Outputs: 
     A function which calculates the dxi/dt's for all species in the simulation.
     Function is in proper format to be utilized by scipy's ode integrate. 
    '''
    # C = coefficient of interaction matrix, size nX(n+1)
    # C = [alpha_1, c11, c12, ..., c1n]
    #     [   .   ,  . ,  . , ..., c2n]
    #     [   .   ,  . ,  . , ...,  . ]
    #     [   .   ,  . ,  . , ...,  . ]
    #     [alpha_n, cn1, c12, ..., cnn]
    # X = column vector of the value of each of the n species
    # X = [x1, x2, ..., xn]
    # Y = [1, x1, x2, ..., xn] 
    # This function returns:
    # X <*> (C*Y)  
    # where * is matrix multiplication and <*> is elementwise multiplication
    return lambda X, t=0: X*array(matrix(C)*matrix(hstack((array([1]),X))).T).reshape(len(X))



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

