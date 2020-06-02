# coding: utf-8
from scipy.integrate import odeint,ode
import numpy as np
from numpy.linalg import solve

"""
----    OpenPyHeat 2020  ----
Solver for 1D multilayer heat transfert equations
------------------------------------------------------------------
Arthur Laffargue

"""

def set_initial_values(T0,size) :

    if isinstance(T0,float) or isinstance(T0,int):
        return np.asarray([T0]*size)
    else :
        if len(T0) != size :
            raise ValueError("Dimension mismatch : size of initial temperatures must be "+str(size))
        else :
            return np.asarray(T0)

def print_odeint_log(t,info):

    args = ('Step','Time','Stepsize','Order')
    print('{0:>5} {1:>15} {2:>15} {3:>12}'.format(*args))
    for i,(ti,dti,o,fail) in enumerate(zip(t,info['hu'],info["mused"],info['tolsf'])) :
            args = (i,round(ti,4),round(dti,4),o,fail)
            print('{0:>5} {1:>15} {2:>15} {3:>12}'.format(*args))

def odeint_scipy_solver(equation_model,T0,timeVec,full_print=False) :
    """
    Solve the equation extracted from the model : equation_model
    T0 is the initial temperature ;
    timeVec  is the times discrete vector ;
    """

    print('-'*50)
    print("""NUMERICAL RESOLUTION OF ODE :
    Solver : Odeint from Scipy ;
    integrator : odeint ;""")

    size = equation_model.get_size()
    invC,K,B,S,A,Ys = equation_model.get_matrix()
    Eq = lambda y,t : invC(y).dot( -K(y).dot(y) + B(y,t) + S(t) + A(y,t) + Ys(y,t) )
    T0 = set_initial_values(T0,size)

    print('Work in progress ... \n')
    if full_print :
        Y,infodict = odeint(Eq,T0,timeVec,full_output=full_print)
        print_odeint_log(timeVec,infodict)
    else :
        Y = odeint(Eq,T0,timeVec)
    print("\nSolution is done.")
    print('-'*50)

    return Y


def lsoda_scipy_solver(equation_model,T0,timeVec,full_print=False) :
    """
    Solve the equation extracted from the model : equation_model
    T0 is the initial temperature ;
    timeVec  is the times discrete vector ;
    """


    print("""NUMERICAL RESOLUTION OF ODE :
    Solver : Ode from Scipy ;
    Integrator : lsoda ;""")

    size = equation_model.get_size()
    invC,K,B,S,A,Ys = equation_model.get_matrix()
    Eq = lambda t,y : invC(y).dot( -K(y).dot(y) + B(y,t) + S(t) + A(y,t) + Ys(y,t) )
    T0 = set_initial_values(T0,size)



    t0 = timeVec[0]
    dt = timeVec[1]-timeVec[0]
    y = np.zeros((len(timeVec),size))
    y[0] = T0

    r =  ode(Eq).set_integrator('lsoda')
    r.set_initial_value(T0, t0)
    i = 0

    print('\nWork in progress ... : ')
    if full_print :
        args = ('Step','Time')
        print('{0:>5} {1:>15} '.format(*args))
    while r.successful() and r.t < timeVec[-1] :
        i+=1
        y[i] = r.integrate(timeVec[i])
        if full_print : print('{0:>5} {1:>15} '.format(i,round(dt*(i+1),4)))
        if np.max(y[i])>5000 :
            print("WARNING : Temperature above 5000K, risk of solver instability.")
    print('Completed')
    print("\nSolution is done.")
    print('-'*50)
    return y





def euler_solver(equation_model,T0,timeVec,full_print=False) :

    print("""NUMERICAL RESOLUTION OF ODE :
    Solver : Euler method ;
    Integrator : semi-implicit Euler solver ;""")


    size = equation_model.get_size()
    invC,K,B,S,A,Ys = equation_model.get_matrix()
    Id = np.eye(size)
    T0 = set_initial_values(T0,size)


    ti = timeVec[0]
    dt = timeVec[1]-timeVec[0]
    Nt = len(timeVec)
    y = np.zeros((len(timeVec),size))
    y[0] = np.array(T0)

    i = 0

    print('\nWork in progress ... : ')
    if full_print :
        args = ('Step','Time')
        print('{0:>5} {1:>15} '.format(*args))

    while i < Nt-1 :
        dt = timeVec[i+1]-timeVec[i]

        invCi = invC(y[i])
        Ki = K(y[i])
        Si = y[i] + dt*invCi.dot( B(y[i],ti) + S(ti) + A(y[i],ti) + Ys(y[i],ti) )
        y[i+1] = solve(Id+dt*invCi.dot(Ki),Si)

        i +=1
        ti += dt

        if full_print : print('{0:>5} {1:>15.5f} '.format(i,ti))
        if np.max(y[i])>5000 :
            print("WARNING : Temperature above 5000K, risk of solver instability.")
    print('Completed')
    print("\nSolution is done.")
    print('-'*50)
    return y

