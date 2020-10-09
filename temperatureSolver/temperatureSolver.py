# coding: utf-8
import numpy as np
from scipy.integrate import odeint,ode
from numpy.linalg import inv, norm
import sys

"""
----    OpenPyHeat 2019  ----
Aide à la résolution d'équation de la thermique en multicouche 1D
------------------------------------------------------------------
Arthur Laffargue
arthur.laffargue@ensam.eu


OpenPyHeat_Solver.py

    - Construction des modèles des couches et des conditions de continuité aux interfaces ;
    - Prise en compte des conditions aux limites ;
    - Construction de l'équation ;
    - Prise en comptes des conditions initiales ;
    - Résolution ; 
"""


class equationModel : 
    
    __solidType = ['Solid','Cylindric']
    __fluidType = ['FluidCavity','FluidFlow',
                   'LeftFluidCavity','LeftFluidFlow',
                   'RightFluidCavity','RightFluidFlow']

    def __init__(self) : 
        self.__Materiaux = [] #Liste des matériaux présents 
        self.__BCs = None #Conditions aux limites 
        self.__Discretisation = [] #Liste des élements de discrétisation
        self.__check = False #Condition pour la construction de l'équation
        self.__Source = [] #Enregistre les sources 
        self.__Interfaces = []#Répertorie les interfaces
        self.__QintList = []
        self.__Advection = []
        self.__method = []
        self.__radiation = []
        self.__symmetry = False
        
        self.__matK = None 
        self.__invMatC = None
        self.__Tsol = []


    def addSolidLayer(self,k,rho,Cp,Nx,e,Qint = None ):
        """Add a solid layer to the study : 
        k : function f(T), is the conductivity. 
        rho : function f(T), is the density. 
        Cp : function f(T), is the heat capacity.
        Nx : int, number of elements.
        e : float, length of the layer.
        Qint : None or function f(x,t), is internal source.

        Units : k(W/(K.m)) ; cp(J/(K.kg)) ; rho  (kg/m3) ; e(m) ;Qint(W/m3)"""


        self.__Materiaux += [['Solid',k,rho,Cp]]
        self.__Discretisation += [[Nx,e/Nx]] 
        if Qint is not None :
            self.__QintList += [Qint]
            As = (np.eye(Nx+1)+np.eye(Nx+1,k=-1))
            As = np.delete(As,Nx,1)
            x0 = np.linspace(0,1,Nx+1)
            x0 = (x0[1:] + x0[:-1])/2
            Qs = lambda t : 0.5*e/Nx*As.dot(Qint(x0,t))
            self.__Source += [[len(self.__Materiaux)-1,Qs]]
        
        self.__method.append('Centered')




    def addCylindricLayer(self,k,rho,Cp,Nx,r1,r2,Qint = None):
        """Add a solid layer to the study : 
        k : function f(T), is the conductivity. 
        rho : function f(T), is the density. 
        Cp : function f(T), is the heat capacity.
        Nx : int, number of elements.
        r1,r2 : floats, the two radius.
        Qint : None or function f(x,t), is internal source.

        Units : k(W/(K.m)) ; cp(J/(K.kg)) ; rho  (kg/m3) ; e(m) ;Qint(W/m3)"""

        ri,re = min(r1,r2),max(r1,r2)
        if ri == re or re<0 or ri<0:
            raise ValueError("Radius must be positive and different and cannot be egal.")
        
        
        if ri == 0 : 
            #La symmétrie est imposé et aucun matériaux ne peut se trouver à gauche
            if len(self.__Materiaux) > 0 : 
                print('ERROR : the internal radius is egal to zero. None layer cannot be present before.')
            print("WARNING : the internal radius is egal to zero : a non-flux conditions should be applied on left side.")    
            self.__symmetry = True
            ri = (re-ri)/Nx
            
        e = re-ri
        self.__Materiaux += [['Cylindric',k,rho,Cp,(ri,re)]]
        self.__Discretisation += [[Nx,e/Nx]] 
        if Qint is not None :
            self.__QintList += [Qint]
            As = (np.eye(Nx+1)+np.eye(Nx+1,k=-1))
            As = np.delete(As,Nx,1)
            # x0 = np.linspace(0,1,Nx+1)
            # x0 = (x0[1:] + x0[:-1])/2
            # Qs = lambda t : 0.5*e/Nx*As.dot(Qint(x0,t))
            xr1 = np.ones(Nx)*ri
            xr2 = np.zeros(Nx)
            for i in range(1,Nx) : 
                xr1[i] = xr1[i-1]+e/Nx
            xr2 = xr1+e/Nx
            x0 = ((xr1+xr2)/2-ri)/(re-ri)
            Qs = lambda t : 0.5*As.dot(Qint(x0,t)*(xr2**2-xr1**2))
            self.__Source += [[len(self.__Materiaux)-1,Qs]]
        
        
        self.__method.append('Coupled')
            
            
            
            
              
    def addFluidCavity(self,rho,Cp,V,S,h,Qint=None):
        """Add a fluid cavity to the study : 
        rho : function f(T), is the density. 
        Cp : function f(T), is the heat capacity.
        V : float, is the total volume of the cavity. 
        S : tuple<float>, the two surfaces of exchanges.
        h : tuple<float>, the two convective exchange coefficients. 
        Qint : None or function f(t), is internal source.

        Units : cp(J/(K.kg)) ; rho  (kg/m3) ; V(m3) ; S(m2) ; h(W/(K.m2) ; Qint(W/m3)"""

        if self.__Materiaux != [] :
            self.__Discretisation += [[2,None]]
            self.__Materiaux += [['FluidCavity',rho,Cp,V,S,h]]
            self.__method.append('Centered')
            if Qint is not None :
                self.__QintList += [Qint]
                Qs = lambda t : [0,V*Qint(t),0]
                self.__Source += [[len(self.__Materiaux)-1,Qs]]
        else : 
            self.__Discretisation += [[1,None]]
            self.__Materiaux += [['LeftFluidCavity',rho,Cp,V,S,h]]
            self.__method.append('Centered')
            if Qint is not None :
                self.__QintList += [Qint]
                Qs = lambda t : [V*Qint(t),0]
                self.__Source += [[len(self.__Materiaux)-1,Qs]]


    def addLiquid0D(self,rho,Cp,V,S,h,Qint=None,qm=None,Tinlet=None):
        """Add a fluid cavity with flow out to the study : 
        rho : function f(T), is the density. 
        Cp : function f(T), is the heat capacity.
        V : float, is the total volume of the cavity. 
        S : tuple<float>, the two surfaces of exchanges.
        h : tuple<float>, the two convective exchange coefficients. 
        Qint : None or function f(t), is internal source.

        Units : cp(J/(K.kg)) ; rho  (kg/m3) ; qv(m3/s) ; Te(K) ;  V(m3) ; S(m2) ; h(W/(K.m2) ; Qint(W/m3)"""

        if self.__Materiaux != [] :
            self.__Discretisation += [[2,None]]
            self.__method.append('Centered')
            self.__Materiaux +=  [['FluidCavity',rho,Cp,V,S,h]]
            if qm is not None : 
                Qinlet = lambda T,t : Cp(0.5*(T+Tinlet(t)))*qm*(Tinlet(t)-T)
                self.__Advection += [[len(self.__Materiaux)-1,Qinlet]]
            if Qint is not None :
                self.__QintList += [Qint]
                Qs = lambda t : [0,V*Qint(t),0]
                self.__Source += [[len(self.__Materiaux)-1,Qs]]
        else : 
            self.__Discretisation += [[1,None]]
            self.__method.append('Centered')
            self.__Materiaux += [['LeftFluidCavity',rho,Cp,V,S,h]]
            if qm is not None : 
                Qinlet = lambda T,t : Cp(0.5*(T+Tinlet(t)))*qm*(Tinlet(t)-T)
                self.__Advection += [[len(self.__Materiaux)-1,Qinlet]]
            if Qint is not None :
                self.__QintList += [Qint]
                Qs = lambda t : [V*Qint(t),0]
                self.__Source += [[len(self.__Materiaux)-1,Qs]]


    def addGas0D(self,rho,Cp,V,S,h,M,Qint=None,qm=None,Tinlet=None):
        """Add a fluid cavity with flow out to the study : 
        rho : function f(T), is the density. 
        Cp : function f(T), is the heat capacity.
        V : float, is the total volume of the cavity. 
        S : tuple<float>, the two surfaces of exchanges.
        h : tuple<float>, the two convective exchange coefficients. 
        Qint : None or function f(t), is internal source.

        Units : cp(J/(K.kg)) ; rho  (kg/m3) ; qv(m3/s) ; Te(K) ;  V(m3) ; S(m2) ; h(W/(K.m2) ; Qint(W/m3)"""
        
        Cv = lambda T : Cp(T) - 8.314/M

        if self.__Materiaux != [] :
            self.__Discretisation += [[2,None]]
            self.__method.append('Centered')
            self.__Materiaux +=  [['FluidCavity',rho,Cv,V,S,h]]
            if qm is not None : 
                Qinlet = lambda T,t : Cp(0.5*(T+Tinlet(t)))*qm*(Tinlet(t)-T)
                self.__Advection += [[len(self.__Materiaux)-1,Qinlet]]
            if Qint is not None :
                self.__QintList += [Qint]
                Qs = lambda t : [0,V*Qint(t),0]
                self.__Source += [[len(self.__Materiaux)-1,Qs]]
        else : 
            self.__Discretisation += [[1,None]]
            self.__method.append('Centered')
            self.__Materiaux += [['LeftFluidCavity',rho,Cv,V,S,h]]
            if qm is not None : 
                Qinlet = lambda T,t : Cp(0.5*(T+Tinlet(t)))*qm*(Tinlet(t)-T)
                self.__Advection += [[len(self.__Materiaux)-1,Qinlet]]
            if Qint is not None :
                self.__QintList += [Qint]
                Qs = lambda t : [V*Qint(t),0]
                self.__Source += [[len(self.__Materiaux)-1,Qs]]
                
                
                
    def addBoundaryConditions(self,Type,params): 
        """Define the external boundaries.
        Type : tuple<string>, describe the type of boundary. 
        params : tuple<function(t)>, functions, flux,temperatures ... depends on the type.
        
        Type = 'Temperature' 
        params : function(t) = Text [K].
        
        Type = 'Flux'
        params : function(t) = Phi_ext [W.m-2].
        
        Type = 'Convection' 
        params : tuple(float1,function2(t)), float1 = h [W/(K.m-2)] ; function2(t) = Text [K]"""

        self.__BCs = [[Type[0],params[0]],[Type[1],params[1]]]


    def addRadiativeTransfert(self,side,emissivity,Text) : 
        """Define an external radiative transfert. 
        side = 'r' or 'l' for right or left ; 
        emissivity : emissity of the grey body ;
        Text : function f(t). External temperature."""
        if self.__BCs != [] : 
            radType = [rad[0] for rad in self.__radiation]
            if side == 'l' : 
                if self.__Materiaux[0][0] not in self.__solidType : 
                    print("WARNING : you cannot define radiative transfert on fluid/gas cavity. Only on solid material.")
                    return 
                    
                if self.__BCs[0][0] == 'Temperature' : 
                    print("WARNING : you cannot define radiative transfert if you impose temperature. Only with flux condition.")
                    return
                
                
                if 'l' in radType : 
                    print('WARNING : you cannot apply two different radiation on the same boundary.')
                    return 
                    
                if self.__Materiaux[0][0] == "Cylindric" : 
                    r1 = self.__Materiaux[0][-1][0]
                else : r1 = 1
                
                self.__radiation += [[0,r1*emissivity,Text]]
                
            if side == 'r' : 
                if self.__Materiaux[-1][0] not in self.__solidType : 
                    print("WARNING : you cannot define radiative transfert on fluid/gas cavity. Only on solid material.")
                    return 
                    
                if self.__BCs[1][0] == 'Temperature' : 
                    print("WARNING : you cannot define radiative transfert if you impose temperature. Only with flux condition.")
                    return
                
                if 'r' in radType : 
                    print('WARNING : you cannot apply two different radiation on the same boundary.')
                    return 
                    
                if self.__Materiaux[-1][0] == "Cylindric" : 
                    r2 = self.__Materiaux[-1][-1][1]
                else : r2 = 1
                
                self.__radiation += [[-1,r2*emissivity,Text]]
        else : 
            print("ERROR : You cannot define radiation before define boundary conditions.")
            return 
            
    ##-------Check préléminaire--------##
               
    def __Check(self) :
        print('-'*50)
        print("VERIFICATION OF THE PARAMETERS : \n...")
        
        #1- Au moins une couche est définie
        if self.__Materiaux == [] :
            raise TypeError("No layer defined.")
            
        #2- Les Conditions aux limites
        elif self.__BCs is None :
            raise TypeError("No boundary conditions defined")
        
        
        #3- Milieu de gauche est un fluide --> condition est un flux
        elif (self.__Materiaux[0][0] in self.__fluidType) and (self.__BCs[0][0] != 'Flux' ):
            raise TypeError('First domain is a fluid/gas/flow domain. You must have a flux boundary condition.')
        
        #4- Milieu de droite est un fluide --> condition est un flux
        elif (self.__Materiaux[-1][0] in self.__fluidType ) and (self.__BCs[-1][0] != 'Flux' ):
            raise TypeError("Last domain is a fluid/gas/flow domain. You must have a flux boundary condition.")
            
        #5- Le fluide à gauche ne peut pas etre seul 
        elif (self.__Materiaux[0][0] == 'LeftFluidFlow'and len(self.__Materiaux) < 2 ) :
            raise TypeError("You cannot apply fluid/gas/flow domain alone.")
        elif (self.__Materiaux[0][0] == 'LeftFluidCavity' and len(self.__Materiaux) < 2 ): 
            raise TypeError("You cannot apply fluid/gas/flow domain alone.")
            
        #Interface
        else :
            #Construction des interfaces
            for i in range(len(self.__Materiaux)-1) :
                self.__Interfaces += [self.__Materiaux[i][0]+'_'+self.__Materiaux[i+1][0]]
            
            Associations = []
            #Différentes associations de fluide possibles 
            for f1 in self.__fluidType : 
                Associations += [f1+'_'+f2 for f2 in self.__fluidType]
                
            Interfaces = set(self.__Interfaces)
            Associations = set(Associations)
            
            
            if Interfaces & Associations != set() : 
                raise TypeError(" You cannot position two fluids side by side. Add a solid layer.")
            else : 
                self.__check = True
                print("Parameters correctly defined")
                print('-'*50)
                
                
        #Fluide à droite sans wall 
        if self.__Materiaux[-1][0] == 'FluidFlow' or self.__Materiaux[-1][0] == 'FluidCavity' :
            self.__Discretisation[-1][0] = 1
            self.__Materiaux[-1][0] = 'Right' + self.__Materiaux[-1][0]
            
            #Source de chaleur 
            if self.__Source != [] : 
                if self.__Source[-1][0] == len(self.__Materiaux)-1 : 
                    if self.__Materiaux[-1][0] == 'RightFluidFlow' : 
                        V = self.__Materiaux[-1][5]
                    if self.__Materiaux[-1][0] == 'RightFluidCavity' : 
                        V = self.__Materiaux[-1][3]
                    self.__Source[-1][1] = lambda t : [0,V*self.__QintList[-1](t)]
                    
    
    
    
    
    ##----------Construction equation -------------#
    def ConstructEquation(self) :
                                          
        try : 
            from .temperatureAssembly import DenseMatrix,RadiativeExchange
        except : 
            from temperatureAssembly import DenseMatrix,RadiativeExchange
            
        self.__Check()
        if not self.__check : return

        print("EQUATION : \n...")
        #Initialisation
        Materiaux = self.__Materiaux
        Interfaces = self.__Interfaces
        BCs = self.__BCs
        Discret = self.__Discretisation
        Sources = self.__Source
        Method = self.__method
        Advection = self.__Advection 
        radiation = self.__radiation
        
        size = sum([Nxi for Nxi,_ in Discret])+1
        self.__size = size


        if Materiaux[0][0] == "Cylindric" : 
            r1 =  2*Materiaux[0][-1][0]
        else : r1 = 1
        
        if Materiaux[-1][0] == "Cylindric" : 
            r2 = 2*Materiaux[-1][-1][1]
        else : r2 = 1
        
        
        invMatC,MatK,VectB,VectS,VectA = DenseMatrix(BCs,
                                                    Materiaux,
                                                    Discret,
                                                    Sources,
                                                    Advection,
                                                    size,
                                                    facteurs =(r1,r2)
                                                    )
        VectYs = RadiativeExchange(radiation,size)
        
        
        T0 = 293*np.ones(size)
        t0 = 0
        C = invMatC(T0)
        K = MatK(T0)
        B = VectB(T0,t0)
        S = VectS(t0)
        Adv = VectA(T0,t0)
        Ys = VectYs(T0,t0)
        try : 
            T0 = 293*np.ones(size)
            t0 = 0
            C = invMatC(T0)
            K = MatK(T0)
            B = VectB(T0,t0)
            S = VectS(t0)
            Adv = VectA(T0,t0)
            Ys = VectYs(T0,t0)
            
            self.__invMatC = invMatC
            self.__matK = MatK
            self.__vectB = VectB
            self.__vectS = VectS
            self.__vectA = VectA
            self.__vectYs = VectYs

            print("Equation successfully defined.")
        except :
            raise ValueError("Equation impossible to build. Check parameters.")

    
    ##-------------Get -------------------------#
    def get_matrix(self):
        return (self.__invMatC,self.__matK,self.__vectB,
                    self.__vectS,self.__vectA,self.__vectYs)
                    
    
    def get_size(self): 
        return self.__size
        
    
    def getMatList(self):
        return self.__Materiaux