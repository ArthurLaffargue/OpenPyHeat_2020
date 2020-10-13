# coding: utf-8
import numpy as np 

def DenseMatrix(BCs,Materiaux,Discret,Sources,Advection,size,facteurs = (1,1)): 

    #Convection
    Indice = [k for k,_,_,_ in Advection]
    indexConvection = []
    for k in Indice:
        if k == 0 : 
            indexConvection.append(0)
        else : 
            N0 = sum([N for N,_ in Discret[:k]]) + 1
            indexConvection.append(N0)
        
    vectA = np.zeros(size)
    def VectA( T,  t ): 
        
        for index,(_,qm,Cp,Tinlet) in zip(indexConvection,Advection) :
            vectA[index] = qm*Cp(T[index])*Tinlet(t)
        return vectA
    
            
    
    
    #Equation aux limites    
    mat0 = Materiaux[0]
    k0 = mat0[1]
    _,dx0 = Discret[0]
    matn = Materiaux[-1]
    kn = matn[1]
    _,dxn = Discret[-1]    
    fact1,fact2 = facteurs 
    

    def invMatC(T) : 
        diagC = np.zeros(size) 
        N0 = 0
        for i,(mat,(Nxi,dx)) in enumerate(zip(Materiaux,Discret)) :
            Tm = T[N0:N0+Nxi+1]
            
            if mat[0] == 'Solid' :
                rho,cp = mat[2],mat[3]
                C = 0.5*rho*cp(0.5*(Tm[1:]+Tm[:-1]))*dx
                
                diagC[N0:N0+Nxi] += C
                diagC[N0+1:N0+Nxi+1] += C
                """
                for i,ci in enumerate(C) :
                    j = N0+i
                    diagC[j:j+2:] += [ci,ci]"""
                          
                   
            if mat[0] == 'Cylindric' : 
                _,k,rho,cp,(ri,re) = mat
                C = 0.5*rho*cp(0.5*(Tm[1:]+Tm[:-1]))
                r1 = np.linspace(ri,re-dx,Nxi)
                r2 = r1+dx
                diagC[N0:N0+Nxi] += C*(r2**2-r1**2)
                diagC[N0+1:N0+Nxi+1] += C*(r2**2-r1**2)
                """
                r1 = ri
                r2 = ri + dx
                C = 0.5*rho*cp(0.5*(Tm[1:]+Tm[:-1]))
                for i,ci in enumerate(C) : 
                    j = N0+i
                    diagC[j:j+2:] += ci*np.array([r2**2-r1**2,r2**2-r1**2])
                    r1 += dx
                    r2 += dx"""
                    
                    
            if mat[0] == 'FluidCavity' : 
                _,rho,cp,V,_,_ = mat
                T1,Tf,T2 = Tm
                c1 = 0.5*V*rho*cp(0.5*(T1+Tf))
                c2 = 0.5*V*rho*cp(0.5*(T2+Tf))
                diagC[N0+1] += c1+c2
                
                
                
            if mat[0] == 'LeftFluidCavity' :
                _,rho,cp,V,_,_ = mat
                T1,T2 = Tm
                diagC[N0] += rho*cp((T1+T2)/2)*V
                
            if mat[0] == 'RightFluidCavity' :
                _,rho,cp,V,_,_ = mat
                T1,T2 = Tm
                diagC[N0+1] += rho*cp((T1+T2)/2)*V
                
            # if mat[0] == 'FluidFlow' : 
            #     _,rho,cp,_,_,V,_,_ = mat
            #     T1,Tf,T2 = Tm
            #     c1 = 0.5*V*rho*cp(0.5*(T1+Tf))
            #     c2 = 0.5*V*rho*cp(0.5*(T2+Tf))
            #     diagC[N0+1] += c1+c2
            # 
            # if mat[0] == 'LeftFluidFlow' :
            #     _,rho,cp,_,_,V,_,_ = mat
            #     T1,T2 = Tm
            #     diagC[N0] += rho*cp((T1+T2)/2)*V
            #     
            # if mat[0] == 'RightFluidFlow' :
            #     _,rho,cp,_,_,V,_,_ = mat
            #     T1,T2 = Tm
            #     diagC[N0+1] += rho*cp((T1+T2)/2)*V                
                                          
            N0+=Nxi     
               
        invMatrixC = np.diag(1/diagC)
        return invMatrixC
        
        
    def MatK(T) :
        matrixK = np.zeros((size,size)) 
        N0 = 0
        for i,(mat,(Nxi,dx)) in enumerate(zip(Materiaux,Discret)) :
            Tm = T[N0:N0+Nxi+1]
            
            if mat[0] == 'Solid' :
                k = mat[1]
                K = 0.5*(k(Tm[1:]) + k(Tm[:-1]))/dx
                
                matrixK[N0:N0+Nxi,N0:N0+Nxi] +=  np.eye(Nxi)*K
                matrixK[N0:N0+Nxi,N0+1:N0+Nxi+1] += -np.eye(Nxi)*K
                matrixK[N0+1:N0+Nxi+1,N0+1:N0+Nxi+1] +=  np.eye(Nxi)*K
                matrixK[N0+1:N0+Nxi+1,N0:N0+Nxi] += -np.eye(Nxi)*K

                """
                for i in range(0,Nxi) :
                    j = N0+i
                    matrixK[j:j+2,j:j+2] += [[K[i],-K[i]],[-K[i],K[i]]]"""
                          
            if mat[0] == 'Cylindric' : 
                _,k,_,_,(ri,re) = mat
                # K = k(Tm)/dx
                K = 0.5*(k(Tm[1:]) + k(Tm[:-1]))/dx
                r1 = np.linspace(ri,re-dx,Nxi)
                r2 = r1+dx
                
                matrixK[N0:N0+Nxi,N0:N0+Nxi] +=  np.eye(Nxi)*K*(r1+r2)
                matrixK[N0:N0+Nxi,N0+1:N0+Nxi+1] += -np.eye(Nxi)*K*(r1+r2)
                matrixK[N0+1:N0+Nxi+1,N0+1:N0+Nxi+1] +=  np.eye(Nxi)*K*(r1+r2)
                matrixK[N0+1:N0+Nxi+1,N0:N0+Nxi] += -np.eye(Nxi)*K*(r1+r2)
                
                """
                r1,r2 = ri,ri+dx
                for i in range(0,Nxi) :
                    j = N0+i
                    matrixK[j:j+2,j:j+2:] += (r1+r2)*np.array([[K[i],-K[i]],[-K[i],K[i]]])
                    r1 += dx
                    r2 += dx"""
                    
                    
            if mat[0] == 'FluidCavity' : 
                _,_,_,_,(S1,S2),(h1,h2) = mat
                T1,Tf,T2 = Tm
                k1 = h1(T1,Tf)
                k2 = h2(Tf,T2)
                r1,r2 = 1.0,1.0
                if Materiaux[i-1][0] == 'Cylindric' :
                    r1 = 2*Materiaux[i-1][-1][1] 
                if Materiaux[i+1][0] == 'Cylindric' :
                    r2 = 2*Materiaux[i+1][-1][0] 
                matrixK[N0:N0+Nxi+1,N0:N0+Nxi+1] += [[k1*r1,-k1*r1,0],
                                                    [-k1*S1,S1*k1+S2*k2,-k2*S2],
                                                    [0,-k2*r2,k2*r2]]    
                                                    
            if mat[0] == 'LeftFluidCavity' : 
                _,_,_,_,S,h = mat
                T1,T2 = Tm
                r = 1.0
                k = h(T1,T2)
                if Materiaux[i+1][0] == 'Cylindric' :
                    r = 2*Materiaux[i+1][-1][0] 
                matrixK[N0:N0+Nxi+1,N0:N0+Nxi+1] += [[S*k,-k*S],
                                                    [-k*r,k*r]]  
                                                    
            if mat[0] == 'RightFluidCavity' : 
                _,_,_,_,S,h = mat
                T1,T2 = Tm
                r = 1.0
                k = h(T1,T2)
                if Materiaux[i-1][0] == 'Cylindric' :
                    r = 2*Materiaux[i-1][-1][1] 
                matrixK[N0:N0+Nxi+1,N0:N0+Nxi+1] += [[r*k,-k*r],
                                                    [-k*S,k*S]]   
                                                    
                                                     
            # if mat[0] == 'FluidFlow' : 
            #     _,rho,cp,_,qv,V,(S1,S2),(h1,h2) = mat
            #     T1,Tf,T2 = Tm
            #     k1 = h1(T1,Tf)
            #     k2 = h2(Tf,T2)
            #     r1,r2 = 1.0,1.0
            #     if Materiaux[i-1][0] == 'Cylindric' :
            #         r1 = 2*Materiaux[i-1][-1][1] 
            #     if Materiaux[i+1][0] == 'Cylindric' :
            #         r2 = 2*Materiaux[i+1][-1][0] 
            #     
            #     matrixK[N0:N0+Nxi+1,N0:N0+Nxi+1] += [[k1*r1,-k1*r1,0],
            #                                         [-k1*S1,S1*k1+S2*k2,-k2*S2],
            #                                         [0,-k2*r2,k2*r2]] 
            #                                         
            #                                         
            # if mat[0] == 'LeftFluidFlow' : 
            #     _,rho,cp,_,qv,V,S,h = mat
            #     T1,T2 = Tm
            #     r = 1.0
            #     k = h(T1,T2)
            #     if Materiaux[i+1][0] == 'Cylindric' :
            #         r = 2*Materiaux[i+1][-1][0] 
            #     print(r)
            #     dfgh
            #     matrixK[N0:N0+Nxi+1,N0:N0+Nxi+1] += h(T1,T2)*np.array([[S,-S],[-r,r]]) 
            #     
            #                                                        
            # if mat[0] == 'RightFluidFlow' : 
            #     _,rho,cp,_,qv,V,S,h = mat
            #     T1,T2 = Tm
            #     r = 1.0
            #     if Materiaux[i-1][0] == 'Cylindric' :
            #         r = 2*Materiaux[i-1][-1][1] 
            #     print(r)
            #     dfgh
            #     matrixK[N0:N0+Nxi+1,N0:N0+Nxi+1] += h(T1,T2)*np.array([[r,-r],[-S,S]])                                              
                                       
            N0+=Nxi              
                          
        if BCs[0][0] == 'Temperature' :
            matrixK[0,0] += k0(T[0])/dx0*1e12
        
        if BCs[1][0] == 'Temperature' :
            matrixK[-1,-1] += kn(T[-1])/dxn*1e12
        
        if BCs[0][0] == 'Convection' : 
            h,Tg = BCs[0][1]
            matrixK[0,0] += h*fact1
            
        if BCs[1][0] == 'Convection' : 
            h,Td = BCs[1][1]
            matrixK[-1,-1] += h*fact2
            
            
        for index,(_,qm,Cp,_) in zip(indexConvection,Advection) :
            matrixK[index,index] += qm*Cp(T[index])
        return matrixK     
        
          
        
    
    
    
    
    ##Boundary conditions     
    def VectB( T, t) : 
        B = np.zeros(size)
        if BCs[0][0] == 'Temperature' : 
            Tg = BCs[0][1]
            B [0] = k0(T[0])*Tg(t)/dx0*1e12
        
        if BCs[0][0] == 'Flux' : 
            phig = BCs[0][1]
            B[0] = phig(t)*fact1
        
        if BCs[0][0] == 'Convection' : 
            h,Tg = BCs[0][1]
            B[0] = h*fact1*Tg(t)
            
            
        if BCs[1][0] == 'Temperature' : 
            Td = BCs[1][1]
            B [-1] = kn(T[-1])*Td(t)/dxn*1e12
        
        if BCs[1][0] == 'Flux' : 
            phid = BCs[1][1]
            B[-1] = phid(t)*fact2
        
        if BCs[1][0] == 'Convection' : 
            h,Td = BCs[1][1]
            B[-1] = h*fact2*Td(t)
        return B
    
    #Sources volumiques
    if Sources == [] : 
        VectS = lambda t : np.zeros(size)
        
    else : 
        def VectS(t) :
            vectS = np.zeros(size)
            for source in Sources :
                i = source[0]
                Qs = source[1]
                Nx,_ = Discret[i]
                N0 = sum([N for N,_ in Discret[:i]])
                vectS[N0:N0+Nx+1] += Qs(t)
            return vectS
    
    
    
    return invMatC,MatK,VectB,VectS,VectA
    
#Transfert Radiatif externes
def RadiativeExchange(radiation,size) : 
    
    if radiation == [] : 
        return lambda T,t : np.zeros_like(T)
        
    else :
        B = np.zeros(size) 
        sigma = 5.67e-8
        def VectR(T,t) : 
            
            for s,e,Text in radiation : 
                B[s] = sigma*e*(Text(t)**4-T[s]**4)
            return B 
        return VectR
    
    
    
    
    
