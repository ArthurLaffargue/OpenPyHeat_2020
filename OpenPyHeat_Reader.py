# coding: utf-8

import os 
import sys
import configparser
from scipy.interpolate import interp1d,interp2d
import time

try : 
    from .utilities.phase_change_material import PCM_properties
    from .utilities.integrate import *
    from .utilities.nonlinearSolver import *
    from .utilities.preProcessing import *
    from .utilities.postProcessing import *
    
    
except : 
    from utilities.phase_change_material import PCM_properties
    from utilities.integrate import *
    from utilities.nonlinearSolver import *
    from utilities.preProcessing import *
    from utilities.postProcessing import *

"""
----    OpenPyHeat 2020  ----
Solver for 1D multilayer heat transfert equations
------------------------------------------------------------------
Arthur Laffargue


"""

__all__ = ["allrun","allclean"]

def OpenData(path) :
    #Ouverture du fichier
    try : 
        data = np.loadtxt(path)
        if data.shape[1] !=2 : 
            raise ValueError("Invalid format for {}. Too much columns, array must be a (N,2) array_like.".format(path))
        if data.shape[0]<2 :
            raise ValueError("Invalid format for {}. Array must be a (N,2) array_like with N > 2.".format(path))
    except :  raise TypeError("Impossible to read {}.".format(path))
    return data
    

def InterpolData(data):
    x = data[:,0]
    y = data[:,1]
    x0,y0 = x[0],y[0]
    xn,yn = x[-1],y[-1]
    
    g = interp1d(x,y,kind='linear',fill_value = (y0,yn),bounds_error=False)
    f = lambda x: y0*(x<=x0)+yn*(x>=xn)+g(x)*(x<xn)*(x>x0)
    return g
    
    
#--------------------------------------------------------------------------------------------------------------------------#
class ReadMat_timesolver : 
    __Solid1DParams = set(['type','kappa','cp','rho','thickness','nel'])
    __Solid0DParams = set(['type','kappa','cp','rho'])
    __FluidCavityParams = set(['type','volume','exchangearea','rho','cp'])
    __GasCavityParams = set(['type','volume','exchangearea','rho','cp','molweight'])
    __PCMParams = set(['type','nel','kappa_s','kappa_l','rho','cp_s','cp_l',
                     'ts','tl','lf'])
    __Cylindric1DParams = set(['type','kappa','cp','rho','r1','r2','nel'])
    __Cylindric0DParams = set(['type','kappa','cp','rho','r1','r2'])
    
    
    __Models = ['solid1D','solid0D','fluidCavity','gasCavity','fluidFlow',
                'PCM','cylindric1D','cylindric0D']
    
    __rPerfectGas = 8.314
    
    
    def __init__(self,path,file = 'Material.txt') : 
        # message initialisation 
        print('-'*50)
        print('READING OF THE MATERIAL PROPERTY DATA INPUTS : ')

        MatList =  [] #Liste des couches telles que définies dans le solveur 
        self.__path = path
        
        #Existence du fichier
        if file not in os.listdir(self.__path) : 
            raise TypeError("No file '"+file+"' found.")
        
        #Ouverture d'un configparser
        else : 
            doc = configparser.ConfigParser()
            doc.read(self.__path+'/'+file)
        
        #Vérification que il y ait bien des couches défines      
        mat_names = doc.sections()
        if mat_names == [] : 
            raise TypeError("No material defined.")
        
        #-----------------------------#
        # Itération sur chaque couche #
        #-----------------------------#
        nCouches = len(mat_names)
        for index,mat in enumerate(mat_names) :
            border  = (index==0) or (index==nCouches-1) 
            items = doc.items(mat)
            Dictionnaire = {key : itm for (key,itm) in items} #Dictionnaire de la couche

            if Dictionnaire["type"] == "solid1D":
                MatList.append( self.__readSolid1D(Dictionnaire,mat) )
                
            elif Dictionnaire["type"] == "solid0D":
                MatList.append( self.__readSolid0D(Dictionnaire,mat) )
                
            elif Dictionnaire["type"] == "cylindric1D":
                MatList.append( self.__readCylindric1D(Dictionnaire,mat) )
                
            elif Dictionnaire['type'] == "fluidCavity" : 
                MatList.append( self.__readFluidCavity(Dictionnaire,mat,border))

            elif Dictionnaire['type'] == "gasCavity" : 
                MatList.append( self.__readGasCavity(Dictionnaire,mat,border) )
                
            elif Dictionnaire['type'] == 'fluidFLow' : 
                MatList.append( self.__readIncompressibleFlow(Dictionnaire,mat,border) )  
                
            elif Dictionnaire['type'] == 'PCM' : 
                MatList.append( self.__readPCM(Dictionnaire,mat) ) 
                
            else : 
                print('ERROR : material '+mat +':')
                print('     Unknown model layer type : ',Dictionnaire['type'])
                print('     Valid models are : ')
                for model in self.__Models : 
                    print('         ',model)
                raise ValueError("Unknown model layer type.")
                                            
        self.MatList =  MatList
        self.MatNames = mat_names
        
        
        
        
        
    def __readSolid1D(self,Dict,mat) : 
        
        #1-   Vérification du paramétrage 
        sF = set(Dict) #Ensemble des parametres fournis 
        sR = self.__Solid1DParams
        sU =  sF & sR
        
        if sU != sR : 
            #Il manque des parametres 
            sD = sR - sU
            print('ERROR : material '+mat +':')
            print('     Parameters : ',sD,' not found.')
            raise ValueError("Missing parameter(s).")
            
        F_k = self.__readK(Dict,mat)
        rho = self.__readRho(Dict,mat)
        F_cp = self.__readCp(Dict,mat)
        Nel = self.__readNel(Dict,mat)
        thickness = self.__readEpaisseur(Dict,mat)
        qint = self.__readSource1D(Dict,mat)

        return ["solid1D",F_k ,rho,F_cp,Nel,thickness,qint]
 
    def __readCylindric1D(self,Dict,mat) : 
        
        #1-   Vérification du paramétrage 
        sF = set(Dict) #Ensemble des parametres fournis 
        sR = self.__Cylindric1DParams
        sU =  sF & sR
        
        if sU != sR : 
            #Il manque des parametres 
            sD = sR - sU
            print('ERROR : material '+mat +':')
            print('     Parameters : ',sD,' not found.')
            raise ValueError("Missing parameter(s).")
        F_k = self.__readK(Dict,mat)
        rho = self.__readRho(Dict,mat)
        F_cp = self.__readCp(Dict,mat)
        Nel = self.__readNel(Dict,mat)
        r1,r2 = self.__readRadius(Dict,mat)
        qint = self.__readSource1D(Dict,mat)

        return ["cylindric1D",F_k ,rho,F_cp,Nel,r1,r2,qint]   

    def __readCylindric0D(self,Dict,mat) : 
        
        #1-   Vérification du paramétrage 
        sF = set(Dict) #Ensemble des parametres fournis 
        sR = self.__Cylindric0DParams
        sU =  sF & sR
        
        if sU != sR : 
            #Il manque des parametres 
            sD = sR - sU
            print('ERROR : material '+mat +':')
            print('     Parameters : ',sD,' not found.')
            raise ValueError("Missing parameter(s).")
        F_k = self.__readK(Dict,mat)
        rho = self.__readRho(Dict,mat)
        F_cp = self.__readCp(Dict,mat)
        Nel = 1
        r1,r2 = self.__readRadius(Dict,mat)
        qint = self.__readSource1D(Dict,mat)

        return ["cylindric0D",F_k ,rho,F_cp,Nel,r1,r2,qint]
        
        
    def __readSolid0D(self,Dict,mat) : 
        
        #1-   Vérification du paramétrage 
        sF = set(Dict) #Ensemble des parametres fournis 
        sR = self.__Solid0DParams
        sU =  sF & sR

        
        if sU != sR : 
            #Il manque des parametres 
            sD = self.__Solid0DParams - sU
            print('ERROR : material '+mat +':')
            print('     Parameters : ',sD,' not found.')
            raise ValueError("Missing parameter(s).")
        F_k = self.__readK(Dict,mat)
        rho = self.__readRho(Dict,mat)
        F_cp = self.__readCp(Dict,mat)
        thickness = self.__readEpaisseur(Dict,mat)
        qint = self.__readSource1D(Dict,mat)

        return ["solid0D",F_k ,rho,F_cp,1,thickness,qint]
    
    
    
    def __readFluidCavity(self,Dict,mat,border=False):
        #1-   Vérification du paramétrage 
        sF = set(Dict) #Ensemble des parametres fournis 
        sR = self.__FluidCavityParams
        sU =  sF & sR
        
        if sU != sR : 
            #Il manque des parametres 
            sD = sR - sU
            print('ERROR : material '+mat +':')
            print('     Parameters : ',sD,' not found.')
            raise ValueError("Missing parameter(s).")
        
        
        rho = self.__readRho(Dict,mat)
        F_cp = self.__readCp(Dict,mat)
        V = self.__readVolume(Dict,mat)
        S = self.__readSurface(Dict,mat,border)
        h = self.__readConvection(Dict,mat,border)
        qint = self.__readSource0D(Dict,mat)
        F_Te,qm = None,None
        
        if 'flow' in Dict : 
            if Dict['flow'].lower() == 'true' : 
                F_Te,qm = self.__readInletFlow(Dict,mat)
        
        
        return ["fluidCavity",rho,F_cp,V,S,h,qint,qm,F_Te]
        
    
    
    def __readGasCavity(self,Dict,mat,border = False):
    
        #1-   Vérification du paramétrage 
        sF = set(Dict) #Ensemble des parametres fournis 
        sR = self.__GasCavityParams
        sU =  sF & sR
        
        if sU != sR : 
            #Il manque des parametres 
            sD = sR - sU
            print('ERROR : material '+mat +':')
            print('     Parameters : ',sD,' not found.')
            raise ValueError("Missing parameter(s).")
        
        
        F_cp = self.__readCp(Dict,mat)
        V = self.__readVolume(Dict,mat)
        S = self.__readSurface(Dict,mat,border)
        h = self.__readConvection(Dict,mat,border)
        qint = self.__readSource0D(Dict,mat)
        molWeight = self.__readMolWeight(Dict,mat)
        rho = self.__readRho(Dict,mat)
        F_Te,qm = None,None
        
        if 'flow' in Dict : 
            if Dict['flow'].lower() == 'true' : 
                F_Te,qm = self.__readInletFlow(Dict,mat)
        
        
        return ["gasCavity",rho,F_cp,V,S,h,qint,molWeight,qm,F_Te]
    
    
    def __readPCM(self,Dict,mat) : 
        #1-   Vérification du paramétrage 
        sF = set(Dict) #Ensemble des parametres fournis 
        sR = self.__PCMParams
        sU =  sF & sR
        
        if sU != sR : 
            #Il manque des parametres 
            sD = sR - sU
            print('ERROR : material '+mat +':')
            print('     Parameters : ',sD,' not found.')
            raise ValueError("Missing parameter(s).")
        
        k_s,k_l = self.__readPCMcond(Dict,mat)    
        rho = self.__readRho(Dict,mat)
        cp_s,cp_l = self.__readPCMcp(Dict,mat)
        Ts,Tl = self.__readPCMtemperature(Dict,mat)
        Lf = self.__readPCMlf(Dict,mat)
        F_k,F_cp = PCM_properties(k_s,k_l,cp_s,cp_l,Lf,Ts,Tl)
        Nel = self.__readNel(Dict,mat)
        qint = self.__readSource1D(Dict,mat)
        geom = self.__readGeom(Dict,mat)

        return ["PCM",F_k ,rho,F_cp,Nel,qint,geom]
        
    
    ##
    #-------------------------------------------------#
    def __readK(self,Dict,mat) :
        #Conductivité
        itm = Dict['kappa'].split()
        if len(itm) != 2 : 
            print("ERROR : invalid syntax : argument 'kappa' in "+mat) 
            raise ValueError("Parameter syntax error.")
            
        if itm[0] == 'constant' :
            try :
                k = float(itm[1].replace(',','.'))
                F_k = lambda T : np.ones_like(T)*k
            except :
                print("ERROR : Float error  --> argument 'kappa' in "+mat) 
                raise ValueError("Parameter floating error.")     
                 
        elif itm[0] == 'data' :
            F_k = InterpolData(OpenData(self.__path + "/data/"+itm[1]+".txt"))
            
        else :  
            print("ERROR : invalid syntax : argument 'kappa' in "+mat) 
            raise ValueError("Parameter syntax error.") 
        return F_k
    
    
    def __readRho(self,Dict,mat) : 
        #rho
            
        itm = Dict['rho'].split()
        if len(itm) != 1 : 
            print("ERROR : invalid syntax : argument 'rho' in  "+mat) 
            raise ValueError("Parameter syntax error.") 
        try :
            rho = float(itm[0].replace(',','.'))
        except :
            print("ERROR : Float error --> argument 'rho' in  "+mat) 
            raise ValueError("Parameter floating error.")     
        return rho
    
    
    
    def __readMolWeight(self,Dict,mat) : 
        #M
            
        itm = Dict['molweight'].split()
        if len(itm) != 1 : 
            print("ERROR : invalid syntax : argument 'molWeight' in  "+mat) 
            raise ValueError("Parameter syntax error.") 
        try :
            molWeight = float(itm[0].replace(',','.'))
        except :
            print("ERROR : Float error --> argument 'molWeight' in  "+mat) 
            raise ValueError("Parameter floating error.")     
        return molWeight
        
        
    def __readCp(self,Dict,mat) :
        #Capacité cp
        itm = Dict['cp'].split()
        if len(itm) != 2 : 
            print("ERROR : invalid syntax : argument 'cp' in "+mat) 
            raise ValueError("Parameter syntax error.")  
            
        if itm[0] == 'constant' :
            try :
                cp = float(itm[1].replace(',','.'))
                F_cp = lambda T : np.ones_like(T)*cp
            except :
                print("ERROR : Float error  --> argument 'cp' in  "+mat) 
                raise ValueError("Parameter floating error.")       
                 
        elif itm[0] == 'data' :
            F_cp = InterpolData(OpenData(self.__path + "/data/"+itm[1]+".txt"))
            
        else :  
            print("ERROR : invalid syntax : argument 'cp' in "+mat) 
            raise ValueError("Parameter syntax error.")
        return F_cp

    def __readInletFlow(self,Dict,mat) :
        #Température inlet
        itm = Dict['tinlet'].split()
        if len(itm) != 2 : 
            print("ERROR : invalid syntax : argument 'Tinlet' in "+mat) 
            raise ValueError("Parameter syntax error.")  
            
        if itm[0] == 'constant' :
            try :
                T_inlet = float(itm[1].replace(',','.'))
                F_Tinlet = lambda T : np.ones_like(T)*T_inlet
            except :
                print("ERROR : Float error  --> argument 'Tinlet' in  "+mat) 
                raise ValueError("Parameter floating error.")    
                 
        elif itm[0] == 'data' :
            F_Tinlet = InterpolData(OpenData(self.__path + "/data/"+itm[1]+".txt"))
            
        else :  
            print("ERROR : invalid syntax : argument 'Tinlet' in "+mat) 
            raise ValueError("Parameter syntax error.")   
            
            
        #Masse flow
        itm = Dict['massflowrate'].split()
        if len(itm) != 1 : 
            print("ERROR : invalid syntax : argument 'massFlowRate' in  "+mat)
            raise ValueError("Parameter syntax error.")
            
        try :
            qm = float(itm[0].replace(',','.'))
        except :
            print("ERROR : Float error --> argument 'massFlowRate' in  "+mat) 
            raise ValueError("Parameter floating error.")          
        return F_Tinlet,qm

            
    def __readEpaisseur(self,Dict,mat) : 
        #Epaisseur
        itm = Dict['thickness'].split()
        if len(itm) != 1 : 
            print("ERROR : invalid syntax : argument 'thickness' in  "+mat) 
            raise ValueError("Parameter syntax error.")
        try :
            thickness = float(itm[0].replace(',','.'))
        except :
            print("ERROR : Float error --> argument 'thickness' in  "+mat) 
            raise ValueError("Parameter floating error.")    
        return thickness
        
    def __readRadius(self,Dict,mat) : 
        #r1
        itm = Dict['r1'].split()
        if len(itm) != 1 : 
            print("ERROR : invalid syntax : argument 'r1' in  "+mat)
            raise ValueError("Parameter syntax error.")
        try :
            r1 = float(itm[0].replace(',','.'))
        except :
            print("ERROR : Float error --> argument 'r1' in  "+mat)
            raise ValueError("Parameter floating error.")
        #r2
        itm = Dict['r2'].split()
        if len(itm) != 1 : 
            print("ERROR : invalid syntax : argument 'r2' in  "+mat)
            raise ValueError("Parameter syntax error.")    
        try :
            r2 = float(itm[0].replace(',','.'))
        except :
            print("ERROR : Float error --> argument 'r2' in  "+mat) 
            raise ValueError("Parameter floating error.")
        return r1,r2
        
    def __readNel(self,Dict,mat) :
        #Nel
        itm = Dict['nel'].split()
        if len(itm) != 1 : 
            print("ERROR : invalid syntax : argument 'Nel' in  "+mat) 
            raise ValueError("Parameter syntax error.") 
        try :
            Nel = int(itm[0].replace(',','.'))
        except :
            print("ERROR : Float error --> argument 'Nel' in  "+mat) 
            raise ValueError("Parameter floating error.")
        return Nel
    
    def __readVolume(self,Dict,mat) : 
        #Volume
            
        itm = Dict['volume'].split()
        if len(itm) != 1 : 
            print("ERROR : invalid syntax : argument 'volume' in  "+mat) 
            raise ValueError("Parameter syntax error.") 
        try :
            V = float(itm[0].replace(',','.'))
        except :
            print("ERROR : Float error --> argument 'volume' in  "+mat) 
            raise ValueError("Parameter floating error.")     
        return V
    
    def __readSurface(self,Dict,mat,border=False) : 
    
        itm = Dict['exchangearea'].split()
        if border : 
            if len(itm) != 1 : 
                print("ERROR : invalid syntax : argument 'exchangeArea' in FluidCavity "+mat) 
                raise ValueError("Parameter syntax error.")
            try :
                S = float(itm[0].replace(',','.'))
                return S
            except :
                print("ERROR : Float error : argument 'exchangeArea' in FluidCavity "+mat)
                raise ValueError("Parameter float error.")        
        
        
        else : 
            if len(itm) != 2 : 
                print("ERROR : invalid syntax : argument 'exchangeArea' in FluidCavity "+mat) 
                raise ValueError("Parameter syntax error.") 
            try :
                S1 = float(itm[0].replace(',','.'))
                S2 = float(itm[1].replace(',','.'))
                return S1,S2
            except :
                print("ERROR : Float error : argument 'exchangeArea' in FluidCavity "+mat)
                raise ValueError("Parameter floating error.")
            
    
    
    def __readConvection(self,Dict,mat,border=False) :
        #h1
        if 'heattransfercoef_1' not in Dict :
            print("ERROR : Missing argument 'heatTransferCoef_1' for FluidCavity "+mat) 
            raise ValueError("Parameter syntax error.")
        itm = Dict['heattransfercoef_1'].split()
        if len(itm) != 2 : 
            print("ERROR : invalid syntax --> argument 'heatTransfertCoef_1' in FluidCavity "+mat)
            raise ValueError("Parameter syntax error.")  
        if itm[0] == 'constant' :
            try :
                h1 = float(itm[1])
                F_h1 = lambda T1,T2 : np.ones_like(T1-T2)*h1
            except :
                print("ERROR : Float error --> argument 'heatTransferCoef_1' in FluidCavity "+mat) 
                raise ValueError("Parameter floating error.")     
        elif itm[0] == 'data' :
            h = InterpolData(OpenData(self.__path + "/data/"+itm[1]+".txt"))
            F_h1 = lambda T1,T2 : h(T1-T2)
    
        if border : return F_h1
        
        else : 
            #h2
            if 'heattransfercoef_2' not in Dict :
                print("ERROR : Missing argument 'heatTransferCoef_2' for FluidCavity "+mat) 
                raise ValueError("Parameter syntax error.")
            itm = Dict['heattransfercoef_2'].split()
            if len(itm) != 2 :
                print("ERROR : invalid syntax --> argument 'heatTransferCoef_2' in FluidCavity "+mat)
                raise ValueError("Parameter syntax error.")  
            if itm[0] == 'constant' :
                try :
                    h2 = float(itm[1])
                    h = lambda T : np.ones_like(T)*h2
                    F_h2 = lambda T1,T2 : h(T1-T2)
                except :
                    print("ERROR : Float error --> argument 'heatTransferCoef_2' in FluidCavity "+mat)
                    raise ValueError("Parameter floating error.") 
            elif itm[0] == 'data' :
                h = InterpolData(OpenData(self.__path + "/data/"+itm[1]+".txt"))
                F_h2 = lambda T1,T2 : h(T1-T2)
            return (F_h1,F_h2)
    
    
    
    def __readSource1D(self,Dict,mat) : 
        #Source volumique 
        if 'heatsource' not in Dict :
            qint = None
        
        else :
            itm = Dict['heatsource']
            if len(itm.split()) != 1 : 
                print("ERROR : invalid syntax : argument 'heatSource' in  "+mat) 
                raise ValueError("Parameter syntax error.") 
            qint = readSourceFile(itm+".txt",self.__path + "/data",dim=1)
        return qint


    def __readSource0D(self,Dict,mat) : 
        #Source volumique 
        if 'heatsource' not in Dict :
            qint = None
        
        else :
            itm = Dict['heatsource']
            if len(itm.split()) != 1 : 
                print("ERROR : invalid syntax : argument 'heatSource' in  "+mat)
                raise ValueError("Parameter syntax error.")    
            qint = readSourceFile(itm+".txt",self.__path + "/data",dim=0)
        return qint


    
    def __readPCMcond(self,Dict,mat) : 
        #Conductivity pcm
        #ks
        itm = Dict['kappa_s'].split()
        if len(itm) != 1 : 
            print("ERROR : invalid syntax : argument 'kappa_s' in  "+mat)
            raise ValueError("Parameter syntax error.")
        try :
            ks = float(itm[0].replace(',','.'))
        except :
            print("ERROR : Float error --> argument 'kappa_s' in  "+mat)
            raise ValueError("Parameter floating error.")
        #kl
        itm = Dict['kappa_l'].split()
        if len(itm) != 1 : 
            print("ERROR : invalid syntax : argument 'kappa_l' in  "+mat)
            raise ValueError("Parameter syntax error.") 
        try :
            kl = float(itm[0].replace(',','.'))
        except :
            print("ERROR : Float error --> argument 'kappa_l' in  "+mat)
            raise ValueError("Parameter floating error.")
        return ks,kl

        
    def __readPCMcp(self,Dict,mat):
        #capacity pcm
        #cps
        itm = Dict['cp_s'].split()
        if len(itm) != 1 : 
            print("ERROR : invalid syntax : argument 'cp_s' in  "+mat)
            raise ValueError("Parameter syntax error.")  
        try :
            cp_s = float(itm[0].replace(',','.'))
        except :
            print("ERROR : Float error --> argument 'cp_s' in  "+mat)
            raise ValueError("Parameter floating error.")
        #cpl
        itm = Dict['cp_l'].split()
        if len(itm) != 1 : 
            print("ERROR : invalid syntax : argument 'cp_l' in  "+mat)
            raise ValueError("Parameter syntax error.")  
        try :
            cp_l = float(itm[0].replace(',','.'))
        except :
            print("ERROR : Float error --> argument 'cp_l' in  "+mat)
            raise ValueError("Parameter floating error.")
        return cp_s,cp_l     


    def __readPCMtemperature(self,Dict,mat):
        #temperature pcm
        #Ts
        itm = Dict['ts'].split()
        if len(itm) != 1 : 
            print("ERROR : invalid syntax : argument 'Ts' in  "+mat) 
            raise ValueError("Parameter syntax error.")   
        try :
            Ts = float(itm[0].replace(',','.'))
        except :
            print("ERROR : Float error --> argument 'Ts' in  "+mat)
            raise ValueError("Parameter floating error.")
        #Tl
        itm = Dict['tl'].split()
        if len(itm) != 1 : 
            print("ERROR : invalid syntax : argument 'Tl' in  "+mat)
            raise ValueError("Parameter syntax error.")
        try :
            Tl = float(itm[0].replace(',','.'))
        except :
            print("ERROR : Float error --> argument 'Tl' in  "+mat)
            raise ValueError("Parameter floating error.")
        return Ts,Tl



    def __readPCMlf(self,Dict,mat):
        #lf
        itm = Dict['lf'].split()
        if len(itm) != 1 : 
            print("ERROR : invalid syntax : argument 'lf' in  "+mat)
            raise ValueError("Parameter syntax error.")   
        try :
            lf = float(itm[0].replace(',','.'))
        except :
            print("ERROR : Float error --> argument 'lf' in  "+mat)
            raise ValueError("Parameter floating error.")
        return lf

    def __readGeom(self,Dict,mat) : 
        if 'geom' in Dict  :
            itm = Dict['geom']
            if itm.lower() == 'cylindric' : 
                geomType = itm.lower()
            elif itm.lower() == 'cartesian' : 
                geomType = itm.lower()
            else : 
                print('ERROR : material '+mat +':')
                print('     Unknown geometry type for PCM : ',Dict['geom'])
                print('     Valid models are : ')
                for model in ["cartesian","cylindric"] : 
                    print('         ',model)
                raise ValueError("Unknown geometry type for PCM.")
        else : 
            geomType = 'cartesian'
            
            
        if geomType == 'cartesian' : 
            if 'thickness' not in Dict :
                print("ERROR : Missing argument 'thikness' for PCM (cartesian) "+mat)
                raise ValueError("Missing argument.")
                
            thickness = self.__readEpaisseur(Dict,mat)
            geom = [geomType,thickness]
        if geomType == 'cylindric' : 
            if 'r1' not in Dict :
                print("ERROR : Missing argument 'r1' for PCM (cartesian) "+mat)
                raise ValueError("Missing argument.")
            if 'r2' not in Dict :
                print("ERROR : Missing argument 'r2' for PCM (cartesian) "+mat)
                raise ValueError("Missing argument.")
            r1,r2 = self.__readRadius(Dict,mat)
            geom = [geomType,r1,r2]
            

        return geom





## Termes sources 
def readSourceFile(file,path,dim=0) : 
    #Existence du fichier
    if file not in os.listdir(path) : 
        print("/!\ ERROR : no file "+file+" found.")
        raise ValueError("Missing file.")
    
    #Ouverture d'un configparser
    else : 
        doc = configparser.ConfigParser()
        doc.read(path+'/'+file)
    
    sections = doc.sections()
    
    if "Parameters" != sections[0] : 
        print("ERROR : no [Parameters] found in ",file)
        raise ValueError("Missing argument.")    

    items = doc.items("Parameters")
    Dict = {key : itm for (key,itm) in items} #Dictionnaire 
        
    if "interpolationscheme" not in Dict : 
        interpScheme = 'linear'
    else : 
        if Dict['interpolationscheme'].lower()  == 'linear' : 
            interpScheme = 'linear'
        elif Dict['interpolationscheme'].lower()  == 'cubic' : 
            interpScheme = 'cubic'
        else : 
            print("ERROR : [Parameters] in ",file," interpolationScheme = ", Dict['interpolationscheme']," not understood")         
            raise ValueError("Parameter syntax error.")
    
    if "timedepend" not in Dict : 
        timeDepend = 0
    else : 
        if Dict['timedepend'].lower()  == 'constant' : 
            timeDepend = 0
        elif Dict['timedepend'].lower()  == 'variable' : 
            timeDepend = 1
        else : 
            print("ERROR : [Parameters] in ",file," timeDepend = ", Dict['timedepend']," not understood") 
            raise ValueError("Parameter syntax error.")
    
    
    if "internalfield" not in Dict : 
        internalField = 0
    if dim != 0 and "internalfield" not in Dict  : 
        print('ERROR : "internalField" not found in ',file)
        raise ValueError("Missing parameter.")
    else : 

        if Dict['internalfield'].lower() == 'uniform' : 
            internalField = 0
        elif Dict['internalfield'].lower()  == 'nonuniform' : 
            internalField = 1
        else : 
            print("ERROR : [Parameters] in ",file," internalField = ", Dict['internalfield']," not understood") 
            raise ValueError("Parameter syntax error.")
    
    
    
    if "Vectors" != sections[1] : 
        print("ERROR : no [Vectors] found in ",file)
        raise ValueError("Missing parameter.")   
    
    items = doc.items("Vectors")
    Dict = {key : itm for (key,itm) in items} #Dictionnaire 
    
    if timeDepend : 
        
        if 'tvector' not in Dict : 
                print('ERROR : "tvector" not found in ',file)
                raise ValueError("Missing parameter.")
        else  : 
            vector = Dict['tvector'].strip('{').strip('}').split()
            try : 
                tvector = [float(s) for s in vector]
            except : 
                print('ERROR : float conversion in "tvector" in ',file)
                raise ValueError("Parameter floating error.")
    
    else : 
        tvector = [0]
    tshape = len(tvector)
    
    if internalField : 
        
        if 'xvector' not in Dict : 
                print('ERROR : "xvector" not found in ',file)
                raise ValueError("Missing parameter.")
        else  : 
            vector = Dict['xvector'].strip('{').strip('}').split()
            try : 
                xvector = [float(s) for s in vector]
            except : 
                print('ERROR : float conversion in "tvector" in ',file)
                raise ValueError("Parameter floating error.")  
    
    else : 
        xvector = [0]
    xshape = len(xvector)
    
    
    
    if 'values' not in Dict : 
        print('ERROR : "values" not found in ',file)
        raise ValueError("Missing parameter.")
    
    values = Dict['values'].strip('{').strip('}')
    lines = list(filter(lambda s: s != '', values.split('\n') ))
    
    if len(lines) != tshape : 
        print("ERROR : 'values' in "+file+" : number of rows doesn't correspond with the shape of time vector")
        raise ValueError("Vector dimensions mismatch.")
    
    sMat = []
    for line in lines : 
        try : 
            sMat.append([float(s) for s in line.strip('(').strip(')').split() ])
        except : 
            print('ERROR : float conversion in "values" in ',file)
            raise ValueError("Parameter floating error.") 
    
    sMat = np.array(sMat)
    
    if interpScheme == 'cubic' :
        if tshape == 1 :
            tvector = [0,1,10,1e9]
            sMat = np.array([sMat[0]]*4)
        if xshape == 1 :
            xvector = [0,0.25,0.75,1.0]
            sMat = np.array([sMat[:,0]]*4).T 
            
    if interpScheme == 'linear' : 
        if tshape == 1 :
            tvector = [0,1e9]
            sMat = np.array([sMat[0]]*2)
        if xshape == 1 :
            xvector = [0,1.0]
            sMat = np.array([sMat[:,0]]*2).T 

    
    return interp2d(xvector, tvector, sMat, kind=interpScheme)






##Boundray condition

def ReadBcsAndInitial(path):
    
    # initialisation 
    print('-'*50)
    print('READING OF THE BOUNDARY CONDITION INPUT : ')
    MatList =  []
    
    doc = configparser.ConfigParser()
    doc.read(path+'/Solver.txt')
    
    #Existence 
    if "Solver.txt" not in os.listdir(path) : 
        print("/!\ ERROR : no file 'Solver.txt' found")
        raise ValueError("File not found.")
    
    #Non vide      
    solver = doc.sections()
    if solver == [] : 
        print("/!\ ERROR : no boundary and initial condition defined ")
        raise ValueError("Missing parameters.")
    
    #LEFT 
    if 'LeftBoundary' not in solver :
        print('ERROR : "LeftBoundary" is missing')
        raise ValueError("Missing parameters.")
        
    items = doc.items('LeftBoundary')
    Dict = {key : itm for (key,itm) in items}

    if Dict['type'] == 'Temperature' :
        if 't' not in Dict :
            print('ERROR : Missing argument T in "LeftBoundary" bc')
            raise ValueError("Missing parameters.")
            
        itm = Dict['t'].split()
        if len(itm) != 2 : 
            print("ERROR : invalid syntax : argument 'T' in LeftBoundary bc")
            raise ValueError("Parameter syntax error.")
               
        if itm[0] == 'constant' :
            try :
                T_g = float(itm[1])
                Fg = lambda t : np.ones_like(t)*T_g
            except :
                print("ERROR : Float error : argument 'T' in LeftBoundary bc")
                raise ValueError("Parameter floating error.")
                       
        elif itm[0] == 'data' :
            Fg = InterpolData(OpenData(path + "/data/"+itm[1]+".txt"))
        BCg = "Temperature"
        
    if Dict['type'] == 'Flux' :
        if 'phi' not in Dict :
            print('ERROR : Missing argument "Phi" in "LeftBoundary" bc')
            raise ValueError("Missing argument.")
        itm = Dict['phi'].split()
        if len(itm) != 2 : 
            print("ERROR : invalid syntax : argument 'Phi' in LeftBoundary bc")
            raise ValueError("Parameter syntax error.")    
        if itm[0] == 'constant' :
            try :
                Phi_g = float(itm[1])
                Fg = lambda t : np.ones_like(t)*Phi_g
            except :
                print("ERROR : Float error : argument 'Phi' in LeftBoundary bc")
                raise ValueError("Parameter floating error.")        
        elif itm[0] == 'data' :
            Fg = InterpolData(OpenData(path + "/data/"+itm[1]+".txt"))
        BCg = "Flux"

    if Dict['type'] == 'Convection' :
        if 't' not in Dict :
            print('ERROR : Missing argument "T" in "LeftBoundary" bc')
            raise ValueError("Missing Parameter.")
        itm = Dict['t'].split()
        if len(itm) != 2 : 
            print("ERROR : invalid syntax : argument 'T' in LeftBoundary bc")
            raise ValueError("Parameter syntax error.")  
        if itm[0] == 'constant' :
            try :
                T_g = float(itm[1])
                Tg = lambda t : np.ones_like(t)*T_g
            except :
                print("ERROR : Float error : argument 'T' in LeftBoundary bc")
                raise ValueError("Parameter floating error.")
                
        elif itm[0] == 'data' :
            Tg = InterpolData(OpenData(path + "/data/"+itm[1]+".txt"))

        if 'h' not in Dict :
            print('ERROR : Missing argument "h" in "LeftBoundary" bc')
            raise ValueError("Missing parameter.")
        itm = Dict['h'].split()
        if len(itm) != 1 : 
            print("ERROR : invalid syntax : argument 'h' in LeftBoundary bc")
            raise ValueError("Parameter syntax error.")
        try :
            hg = float(itm[0])
        except : 
            print("ERROR : Float error : argument 'h' in LeftBoundary bc")
            raise ValueError("Parameter floating error.")    
        BCg = "Convection"
        Fg = (hg,Tg)
        
    #Radiation
    radiationLeft = None
    if 'radiation' in Dict : 
        if Dict['radiation'] == 'on' : 
            print('Radiation ON left side')
            radiationLeft = []
            
            if 'emissivity' not in Dict : 
                print('ERROR : Missing argument "emissivity" in "LeftBoundary" bc')
                raise ValueError("Missing argument.")
            itm = Dict['emissivity'].split()
            if len(itm) != 1 : 
                print("ERROR : invalid syntax : argument 'emissivity' in LeftBoundary bc")
                raise ValueError("Parameter syntax error.")
            try :
                emissivityLeft = float(itm[0])
            except : 
                print("ERROR : Float error : argument 'emissivity' in LeftBoundary bc")
                raise ValueError("Parameter floating error.")       

            if 'tradiation' not in Dict :
                print('ERROR : Missing argument "Tradiation" in "LeftBoundary" bc')
                raise ValueError("Missing argument.")
            itm = Dict['tradiation'].split()
            if len(itm) != 2 : 
                print("ERROR : invalid syntax : argument 'Tradiation' in LeftBoundary bc")
                raise ValueError("Parameter syntax error.")
            if itm[0] == 'constant' :
                try :
                    TradLeft = float(itm[1])
                    TradiationLeft = lambda t : np.ones_like(t)*TradLeft
                except :
                    print("ERROR : Float error : argument 'Tradiation' in LeftBoundary bc")
                    raise ValueError('Parameter floating error.')       
            elif itm[0] == 'data' :
                TradiationLeft = InterpolData(OpenData(path + "/data/"+itm[1]+".txt"))
                
            radiationLeft = ['l',emissivityLeft,TradiationLeft]
        else : 
            print('Radiation OFF on LeftBoundary side')
        
    
    #RIGHT 
    if 'RightBoundary' not in solver :
        print('ERROR : "RightBoundary" is missing')
        raise ValueError("Missing section.")
    items = doc.items('RightBoundary')
    Dict = {key : itm for (key,itm) in items}

    if Dict['type'] == 'Temperature' :
        if 't' not in Dict :
            print('ERROR : Missing argument T in "RightBoundary" bc')
            raise ValueError("Missing parameter.")
        itm = Dict['t'].split()
        if len(itm) != 2 : 
            print("ERROR : invalid syntax : argument 'T' in RightBoundary bc")
            raise ValueError("Parameter syntax error.")
        if itm[0] == 'constant' :
            try :
                T_d = float(itm[1])
                Fd = lambda t : np.ones_like(t)*T_d
            except :
                print("ERROR : Float error : argument 'T' in RightBoundary bc") 
                raise ValueError("Parameter floating error.")    
        elif itm[0] == 'data' :
            Fd = InterpolData(OpenData(path + "/data/"+itm[1]+".txt"))
        BCd = "Temperature"
        
    if Dict['type'] == 'Flux' :
        if 'phi' not in Dict :
            print('ERROR : Missing argument "Phi" in RightBoundary bc')
            raise ValueError("Missing parameter.")
        itm = Dict['phi'].split()
        if len(itm) != 2 : 
            print("ERROR : invalid syntax : argument 'phi' in RightBoundary bc")
            raise ValueError("Parameter syntax error.")  
        if itm[0] == 'constant' :
            try :
                Phi_d = float(itm[1])
                Fd = lambda t : np.ones_like(t)*Phi_d
            except :
                print("ERROR : Float error : argument 'phi' in RightBoundary bc")
                raise ValueError("Parameter floating error.")       
        elif itm[0] == 'data' :
            Fd = InterpolData(OpenData(path + "/data/"+itm[1]+".txt"))
        BCd = "Flux"

    if Dict['type'] == 'Convection' :
        if 't' not in Dict :
            print('ERROR : Missing argument "T" in RightBoundary bc')
            raise ValueError("Missing parameter.")
        itm = Dict['t'].split()
        if len(itm) != 2 : 
            print("ERROR : invalid syntax : argument 'T' in RightBoundary bc")
            raise ValueError("Parameter syntax error.")
        if itm[0] == 'constant' :
            try :
                T_d = float(itm[1])
                Td = lambda t : np.ones_like(t)*T_d
            except :
                print("ERROR : Float error : argument 'T' in RightBoundary bc") 
                raise ValueError("Parameter floating error.")      
        elif itm[0] == 'data' :
            Td = InterpolData(OpenData(path + "/data/"+itm[1]+".txt"))

        if 'h' not in Dict :
            print('ERROR : Missing argument "h" in RightBoundary bc')
            raise ValueError("Missing parameter.")
        itm = Dict['h'].split()
        if len(itm) != 1 : 
            print("ERROR : invalid syntax : argument 'h' in RightBoundary bc")
            raise ValueError("Parameter syntax error.")
        try :
            h = float(itm[0])
        except : 
            print("ERROR : Float error : argument 'h' in RightBoundary bc")
            raise ValueError("Parameter floating error.")
        BCd = "Convection"
        Fd = (h,Td)


    #Radiation
    radiationRight = None
    if 'radiation' in Dict : 
        if Dict['radiation'] == 'on' : 
            print('Radiation ON right side')
            radiationRight = []
            
            if 'emissivity' not in Dict : 
                print('ERROR : Missing argument "emissivity" in RightBoundary bc')
                raise ValueError("Missing parameter.")
            itm = Dict['emissivity'].split()
            if len(itm) != 1 : 
                print("ERROR : invalid syntax : argument 'emissivity' in RightBoundary bc")
                raise ValueError("Parameter syntax error.")
            try :
                emissivityRight = float(itm[0])
            except : 
                print("ERROR : Float error : argument 'emissivity' in RightBoundary bc")
                raise ValueError("Parameter floating error.")        

            if 'tradiation' not in Dict :
                print('ERROR : Missing argument "Tradiation" in RightBoundary bc')
                raise ValueError("Missing parameter.")
            itm = Dict['tradiation'].split()
            if len(itm) != 2 : 
                print("ERROR : invalid syntax : argument 'Tradiation' in RightBoundary bc")
                raise ValueError("Parameter syntax error.") 
            if itm[0] == 'constant' :
                try :
                    TradRight = float(itm[1])
                    TradiationRight = lambda t : np.ones_like(t)*TradRight
                except :
                    print("ERROR : Float error : argument 'Tradiation' in RightBoundary bc") 
                    raise ValueError("Parameter floating error.")      
            elif itm[0] == 'data' :
                TradiationRight = InterpolData(OpenData(path + "/data/"+itm[1]+".txt"))
                
            radiationRight = ['l',emissivityRight,TradiationRight]
        else : 
            print('Radiation OFF on right side')

    print("\nReading completed successfully")
    
    
    print('-'*50)
    print('READING OF THE BOUNDARY CONDITION INPUT : ')
    
    if 'Initial' not in solver :
        print('ERROR : "Initial" is missing')
        raise ValueError("Missing section.")
    items = doc.items('Initial')
    Dict = {key : itm for (key,itm) in items}
    
    
    if 't' not in Dict :
        print('ERROR : Missing argument T in "Initial"')
        raise ValueError("Missing parameter.")
    itm = Dict['t'].split()

    try :
        T0 = float(itm[0])
    except :
        print("ERROR : Float error : argument 'T' in Initial")
        raise ValueError("Parameter floating error.") 

    
    return (BCg,BCd),(Fg,Fd),(radiationLeft,radiationRight),T0







##SOLVER

def ReadSolver(path) :
    
    # initialisation 
    print('-'*50)
    print('READING OF THE SOLVER OPTIONS INPUT : ')
    
    doc = configparser.ConfigParser()
    doc.read(path+'/Solver.txt')
    
    #Existence
    if "Solver.txt" not in os.listdir(path) : 
        print("/!\ ERROR : no file 'Solver.txt' found")
        raise ValueError("Missing file.")
    
    #Non vide      
    Solver = doc.sections()
    if Solver == [] : 
        print("/!\ ERROR : no solver options defined ")
        raise ValueError("Missing section.")
        
        
    #Solveur
    if 'Solver' not in Solver :
        print('ERROR : Missing "Solver" options')
        raise ValueError("Missing section.")
    items = doc.items('Solver')
    Dict = {key : itm for (key,itm) in items}
    
    
    #Application
    if 'application' not in Dict :
        print('ERROR : Missing argument "application".')
        raise ValueError("Missing parameter.")
    application = Dict['application'].split()[0]
    if application not in ["transient","steadyState"] :
        print('ERROR : solver configuration')
        print('     Unknown application : ',application)
        print('     Valid applications are : ')
        for model in ["transient","steadyState"]  : 
            print('         ',model)
        raise ValueError("Unknown application.")
    
    #solvers
    if application == "transient" : 
        print('     -application : transient simulation ; ')
        return 'transient',readTransientSolver(Dict,Solver,doc)
        
    if application == "steadyState" : 
        print('     -application : steady simulation ; ')
        return 'steadyState',readSteadyStateSolver(Dict,Solver,doc)
    
    
def readTransientSolver(Dict,Solver,doc):    

    integrator_list = ["Odeint","Lsoda","Euler"]
    model_list = ["enthalpy","temperature"]
    
    #integrator
    if 'solver' not in Dict :
        print('ERROR : Missing argument "solver".')
        raise ValueError("Missing parameter.")
    solver = Dict['solver'].split()[0]
    if solver not in integrator_list :
        print('ERROR : solver configuration')
        print('     Unknown solver : ',solver)
        print('     Valid solver are : ')
        for model in integrator_list : 
            print('         ',model)
        raise ValueError("Unknown solver type.")
       
    #model thermo
    if 'thermo' not in Dict :
        print('ERROR : Missing argument "thermo".')
        raise ValueError("Missing parameter.")
    thermo = Dict['thermo'].split()[0]
    if thermo not in model_list :
        print('ERROR : solver configuration')
        print('     Unknown thermo model : ',thermo)
        print('     Valid thermo model are : ')
        for model in model_list : 
            print('         ',model)
        raise ValueError("Unknown thermo model.")    
        
    print('     -thermo. eq. variable :',thermo,' ; ')
    print('     -Num. integration :',solver,' ; ')
        
    if 'print' not in Dict :
        full_print = False

    else : 
        printOpt = Dict['print'].split()[0]
        if printOpt.lower() == 'true' : 
            full_print = True
        else : 
            full_print = False

    #Vecteur Temps
    if 'Time' not in Solver :
        print('ERROR : Missing "Time" options') ; sys.exit()
    items = doc.items('Time')
    Dict = {key : itm for (key,itm) in items}
    #---
    if 't0' not in Dict :
        print('ERROR : Missing argument "t0".') ; sys.exit()
    itm = Dict['t0'].split()
    try :
        t0 = float(itm[0])
    except : 
        print("ERROR : Float error : argument 't0'.") ; sys.exit()

    #---
    if 'tf' not in Dict :
        print('ERROR : Missing argument "tf".') ; sys.exit()
    itm = Dict['tf'].split()
    try :
        tf = float(itm[0])
    except : 
        print("ERROR : Float error : argument 'tf'.") ; sys.exit()
    if tf<=t0 :
        print("ERROR : tf must be upper than t0.") ; sys.exit()

    
    if 'nt' not in Dict :
        print('ERROR : Missing argument "Nt".') ; sys.exit()
    itm = Dict['nt'].split()
    try :
        Nt = int(itm[0])
    except : 
        print("ERROR : Float error : argument 'Nt'.") ; sys.exit()
    if Nt<2 :
        print("ERROR : Nt must be upper than 2.") ; sys.exit()

    if 'write_step' not in Dict :
        wstep = 1
    else :
        itm = Dict['write_step'].split()
        try :
            wstep = int(itm[0])
        except : 
            print("ERROR : Float error : argument 'write_step'.") ; sys.exit()

    Xt = np.linspace(t0,tf,Nt)

    return Xt,solver,wstep,full_print
    
    
def readSteadyStateSolver(Dict,Solver,doc):    

    solver_list = ["ScipyRoot","NewtonKrylov","ImplicitLinear","BFGS"]
    
    #Solver
    if 'solver' not in Dict :
        print('ERROR : Missing argument "solver".')
        raise ValueError("Missing parameter.")
    solver = Dict['solver'].split()[0]
    if solver not in solver_list :
        print('ERROR : solver configuration')
        print('     Unknown solver : ',solver)
        print('     Valid solver are : ')
        for model in solver_list : 
            print('         ',model)
        raise ValueError("Unknown solver type.")
       

    print('     -Non-lin. solver :',solver,' ; ')
    
    
    
    
    
        
    if 'print' not in Dict :
        full_print = False

    else : 
        printOpt = Dict['print'].split()[0]
        if printOpt.lower() == 'true' : 
            full_print = True
        else : 
            full_print = False


    return solver,full_print

##ALLRUN ALLCLEAN
def allclean(path):
    cleanCase(path)
    
    
def allrun(path):
    try : from .temperatureSolver.temperatureSolver import equationModel
    except : from temperatureSolver.temperatureSolver import equationModel
    
    #Cleaning
    cleanCase(path)
    
    #Reading
    reader = ReadMat_timesolver(path)
    BCs,BCsFunc,(radiationL,radiationR),T0 = ReadBcsAndInitial(path)
    application,solver_option = ReadSolver(path)
    
    matList = reader.MatList
    Eq = equationModel()
    
    #Set up equation model
    for mat in matList : 
        
        if mat[0] == 'solid1D' or mat[0] == "solid0D": 
            _,k,rho,cp,np,e,qint = mat
            Eq.addSolidLayer(k,rho,cp,np,e,qint)
        
        if mat[0] == "cylindric1D" or mat[0] == "cylindric0D" : 
            _,k,rho,cp,nel,r1,r2,qint = mat
            Eq.addCylindricLayer(k,rho,cp,nel,r1,r2,qint)
            
        if mat[0] == 'fluidCavity' : 
            _,rho,cp,v,s,h,qint,qm,Tinlet = mat
            Eq.addLiquid0D(rho,cp,v,s,h,qint,qm,Tinlet)
        
        if mat[0] == 'gasCavity' : 
            _,rho,cp,v,s,h,qint,M,qm,Tinlet = mat
            Eq.addGas0D(rho,cp,v,s,h,M,qint,qm,Tinlet)
            
        if mat[0] == 'PCM' : 
            _,k ,rho,cp,np,qint,geom = mat 
            if geom[0] == 'cartesian': 
                e = geom[1]
                Eq.addSolidLayer(k,rho,cp,np,e,qint)
            if geom[0] == 'cylindric':
                r1,r2 = geom[1],geom[2]
                Eq.addCylindricLayer(k,rho,cp,np,r1,r2,qint)
                
                
    Eq.addBoundaryConditions(BCs,BCsFunc)
    if radiationL is not None : 
        Eq.addRadiativeTransfert(radiationL[0],radiationL[1],radiationL[2])
    if radiationR is not None : 
        Eq.addRadiativeTransfert(radiationR[0],radiationR[1],radiationR[2])
    Eq.ConstructEquation()
    
    #Solve
    time_comp_start = time.time()
    if application == 'transient' : 
        Xt,solver,wstep,full_print = solver_option
        if solver == "Odeint" :         
            Ysol = odeint_scipy_solver(Eq,T0,Xt,full_print=full_print)
            
        if solver == 'Lsoda' : 
            Ysol = lsoda_scipy_solver(Eq,T0,Xt,full_print=full_print)
            
        if solver == 'Euler' : 
            Ysol = euler_solver(Eq,T0,Xt,full_print=full_print)
    
    if application == 'steadyState' : 
        solver,full_print = solver_option 
        if solver == 'ScipyRoot' : 
            Ysol = ScipyRoot(Eq,T0,full_print)
        
        if solver == 'NewtonKrylov' : 
            Ysol = NewtonKrylov(Eq,T0,full_print)
        
        if solver == 'ImplicitLinear' : 
            Ysol = SemiImplicitLinearSolver(Eq,T0,full_print)
            
        if solver == 'BFGS' : 
            Ysol = BFGSSolver(Eq,T0,full_print)
    time_comp_end = time.time()
    print("Computation time : %.3f s"%(time_comp_end-time_comp_start))
    
    
    #POST PROC
    matNames = reader.MatNames
    matNodeList,matNodePos = buildMesh(path,matList,matNames)
    
    
    
    if application == 'transient' :
        Ysol,Xt = Ysol[::wstep],Xt[::wstep] 
        shape = (len(Xt),len(Ysol[0]))
        saveTransientSolution(path,Xt,Ysol,shape)
        saveMaterialSolutionTransient(path,matNodeList,matList,matNames,Xt,Ysol)
        saveFinalSolution(path,Ysol[-1])
        plotTransient(matNames,matList,matNodeList,Xt,Ysol,path) 
        
        return Ysol,Xt
    
    if application == 'steadyState' : 
        saveMaterialSolutionSteady(path,matNodeList,matList,matNames,Ysol)
        saveFinalSolution(path,Ysol)
        plotSteady(matNames,matList,matNodeList,Ysol,path) 
        return Ysol
    


    
