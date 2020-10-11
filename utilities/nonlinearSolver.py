# coding: utf-8
from scipy.optimize import root
import numpy as np
from numpy.linalg import solve,norm


"""
----    OpenPyHeat 2020  ----
Solver for 1D multilayer heat transfert equations
------------------------------------------------------------------
Arthur Laffargue

"""
#
# def norm(X) :
#     return np.sqrt(np.sum(X**2))


def set_initial_values(T0,size) :

    if isinstance(T0,float) or isinstance(T0,int):
        return np.asarray([T0]*size,dtype=float)
    else :
        if len(T0) != size :
            raise ValueError("Dimension mismatch : size of initial temperatures must be "+str(size))
        else :
            return np.asarray(T0,dtype=float)


def ScipyRoot(equation_model,T0,full_print=False) :
    """
    Solve the equation extracted from the model : equation_model
    T0 is the initial temperature ;
    """

    print('-'*50)
    print("""NUMERICAL RESOLUTION OF NON-LINEAR EQUATIONS :
    Solver : Scipy root with modified Powell method (default solver) ;""")

    size = equation_model.get_size()
    invC,K,B,S,A,Ys = equation_model.get_matrix()
    Eq = lambda y : ( -K(y).dot(y) + B(y,0) + S(0) + A(y,0) + Ys(y,0) )
    T0 = set_initial_values(T0,size)

    print('Work in progress ... \n')
    sol = root(Eq,T0,method='hybr')

    print("Success : ",bool(sol.success))
    if full_print : print(sol.message)
    print("\nSolution is done.")



    print('-'*50)

    return sol.x


def NewtonKrylov(equation_model,T0,full_print=False) :
    """
    Solve the equation extracted from the model : equation_model
    T0 is the initial temperature ;
    """

    print('-'*50)
    print("""NUMERICAL RESOLUTION OF NON-LINEAR EQUATIONS :
    Solver : Newton Krylov (scipy) with LMGRES ;""")

    size = equation_model.get_size()
    invC,K,B,S,A,Ys = equation_model.get_matrix()
    Eq = lambda y : ( -K(y).dot(y) + B(y,0) + S(0) + A(y,0) + Ys(y,0) )
    T0 = set_initial_values(T0,size)

    print('Work in progress ... \n')
    sol = root(Eq,T0,method='krylov')

    print("Success : ",bool(sol.success))
    if full_print : print(sol.message)
    print("\nSolution is done.")



    print('-'*50)

    return sol.x


def SemiImplicitLinearSolver(equation_model,T0,full_print=False,maxiter=350,tol=1e-6,relax=0.95):
    """
    Solve the equation extracted from the model : equation_model
    T0 is the initial temperature ;
    """

    print('-'*50)
    print("""NUMERICAL RESOLUTION OF NON-LINEAR EQUATIONS :
    Solver : Semi Implicit Iterative Method ;""")

    size = equation_model.get_size()
    invC,K,B,S,A,Ys = equation_model.get_matrix()
    Q = lambda y : (  B(y,0) + S(0) + A(y,0) + Ys(y,0) )
    T0 = set_initial_values(T0,size)

    print('Work in progress ... \n')



    K0,Q0 = K(T0),Q(T0)
    r0 = norm(K0.dot(T0) - Q0)/size
    dT = 1.0
    err = 1.0
    iter = 0

    while (iter<maxiter)&(dT>tol)&(err>tol) :
        T1 = solve(K0,Q0)
        dT = norm(T1-T0)/norm(T0)
        T0 = relax*T1 + (1-relax)*T0
        K0 = K(T0)
        Q0 = Q(T0)
        r = norm(K0.dot(T0)-Q0)/size
        err = r/r0
        iter += 1
        if full_print :
            print("ITERATION : ",iter,
                "RESIDUAL %.3e "%err,"||dT|| %.3e"%dT)

    print("\nSolution is done.")



    print('-'*50)

    return T0


def BFGSSolver(equation_model,T0,full_print=False,maxiter=350,tol=1e-6,relax=0.95):
    """
    Solve the equation extracted from the model : equation_model
    T0 is the initial temperature ;
    """

    print('-'*50)
    print("""NUMERICAL RESOLUTION OF NON-LINEAR EQUATIONS :
    Solver : Semi Implicit Iterative Method ;""")

    size = equation_model.get_size()
    invC,K,B,S,A,Ys = equation_model.get_matrix()
    Q = lambda y : (  B(y,0) + S(0) + A(y,0) + Ys(y,0) )
    T0 = set_initial_values(T0,size)

    print('Work in progress ... \n')

    K0,Q0 = K(T0),Q(T0)
    Hmat = K0
    F0 = K0.dot(T0) - Q0
    r0 = norm(F0)/size
    dT = 1.0
    err = 1.0
    iter = 0

    df = np.zeros((size,1))
    sk = np.zeros((size,1))

    while (iter<maxiter)&(dT>tol)&(err>tol) :
        pk = solve(Hmat,-F0)

        a = K0.dot(T0)
        b = K0.dot(pk)
        c = -Q0
        m = np.sum(b**2)
        n = np.sum(2*a*b+2*c*b)
        alpha = -1/2*n/m

        sk[:,0] =  alpha*pk
        T1 = T0 +  alpha*pk
        K1,Q1 = K(T1),Q(T1)
        F1 = K1.dot(T1)-Q1


        df[:,0] = F1-F0
        skT = sk.T
        dfT = df.T

        Uk = (df.dot(dfT))/(dfT.dot(sk))
        Vk = -(Hmat.dot(sk).dot(skT.dot(Hmat)))/(skT.dot(Hmat.dot(sk)))
        Hmat = Hmat + Uk + Vk


        dT = norm(T1-T0)/norm(T0)
        T0 = T1
        K0 = K1
        Q0 = Q1

        r = norm(K0.dot(T0)-Q0)/size
        err = r/r0
        iter += 1
        if full_print :
            print("ITERATION : ",iter,
                "RESIDUAL %.3e "%err,"||dT|| %.3e"%dT)

    print("\nSolution is done.")



    print('-'*50)

    return T0