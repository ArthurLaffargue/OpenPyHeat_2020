# coding: utf-8
import numpy as np

"""
----    OpenPyHeat 2020  ----
Solver for 1D multilayer heat transfert equations
------------------------------------------------------------------
Arthur Laffargue


phase_change_material.py

    - Interpolation and extrapolation of solid and liquid properties ;
    - Interpolation and extrapolation of enthalpy jump inside an equivalent heat capacity;
"""


def PCM_properties(ks,kl,cps,cpl,Lf,Ts,Tl) :
    """
    ks,kl : solid and liquid conductivity ;
    rhos,rhol : liquid and solid density ;
    cps,cpl : liquid and solid heat capacity ;
    Lf : Latent heat of fusion ;
    Ts,Tl : Temperature of melting and solidication points  ;
    """


    #---- Continuité de la conductivité -----#
    #Ecriture du système
    A=np.zeros((4,4))
    A[0]=[Ts**i for i in range(4)]
    A[1]=[Tl**i for i in range(4)]
    A[2]=[i*Ts**(i-1) for i in range(4)]
    A[3]=[i*Tl**(i-1) for i in range(4)]

    B=np.array([ks,kl,0,0])
    coef_k=np.linalg.solve(A,B)

    #Ecriture de la fonction
    ki= lambda x : sum([c*x**i for i,c in enumerate(coef_k)])
    k = lambda x : ki(x)*(Ts<=x)*(x<=Tl)+ks*(x<Ts)+kl*(x>Tl) #Fonction finale

    #---- Continuité de la chaleur spécifique -----#
    #Ecriture du système
    A=np.zeros((5,5))
    A[0]=[Ts**i for i in range(5)]
    A[1]=[Tl**i for i in range(5)]
    A[2]=[i*Ts**(i-1) for i in range(5)]
    A[3]=[i*Tl**(i-1) for i in range(5)]
    A[4]=[(Tl**i-Ts**i)/i for i in range(1,6)]

    B=np.array([cps,cpl,0,0,Lf+cps*Ts])
    coef=np.linalg.solve(A,B)

    cpi = lambda x : sum([c*x**i for i,c in enumerate(coef)])

    cp = lambda x : cpi(x)*(Ts<x)*(x<Tl)+cps*(x<=Ts)+cpl*(x>=Tl) #Fonction finale

    return k,cp



##
#TEST
if __name__ == '__main__' :
    import matplotlib.pyplot as plt

    k,cp = PCM_properties(1.6,0.6,2060,4185,333e3,270.5,275.5)


    T = np.linspace(265,285,2500)

    plt.style.use('ggplot')
    plt.figure(1)
    plt.subplot(121)
    plt.plot(T,k(T),color='crimson')
    plt.title("Conductivité",fontsize=11)
    plt.ylabel("k(T)")
    plt.xlabel("Température K")
    plt.margins(0.1)
    plt.subplot(122)
    plt.plot(T,cp(T),color='g')
    plt.title("Capacité thermique apparente",fontsize=11)
    plt.ylabel("C_p(T)")
    plt.xlabel("Température K")
    plt.margins(0.1)
    plt.subplots_adjust(hspace=0.,wspace=0.8)






    plt.grid(True)
    plt.show()

