import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib import colors as mcolors
from cycler import cycler
import os
try :
    plt.style.use('bmh')
    plt.rcParams["font.family"] = "serif"
except : pass


def postProc_main(path,matList,Tsol) :


    return None


def saveTransientSolution(path,Xt,Tsol,shape,name='solution.txt'):

    Tsol = Tsol.reshape(shape)
    shape = (shape[0],shape[1]+1)
    solution = np.zeros(shape)
    solution[:,0] = Xt
    solution[:,1:] = Tsol
    header = '-'*50+'\n'
    header += '     SOLUTION'+'\n'
    header += '\n'
    header += '-'*50+'\n'
    header += '\n'
    header += '|time|node 0|...|node n'+'|\n'

    np.savetxt(path+'/solution/'+name,solution,header=header,fmt='%.5e')


def saveSteadySolution(path,Tsol,name='solution.txt'):


    header = '-'*50+'\n'
    header += '     SOLUTION'+'\n'
    header += '\n'
    header += '-'*50+'\n'
    header += '\n'
    header += '|node 0|...|node n'+'|\n'

    np.savetxt(path+'/solution/'+name,np.array([Tsol]),header=header,fmt='%.5e')

def saveMaterialSolutionTransient(path,matNodeList,matList,matNames,Xt,Xsol):

    for name,node,mat in zip(matNames,matNodeList,matList) :
        if  mat[0] == 'solid1D' or mat[0] == "cylindric1D" or mat[0]=='PCM':
            Tsol = Xsol[:,node]
            shape = (len(Xt),len(node))
            saveTransientSolution(path,Xt,Tsol,shape,name=name+'_T.txt')

        if mat[0]=="fluidCavity" :
            Tsol = Xsol[:,node]
            shape = (len(Xt),len(node))
            saveTransientSolution(path,Xt,Tsol,shape,name=name+'_T.txt')

        if mat[0] == 'gasCavity' :
            _,rho,cp,v,s,h,qint,M,qm,Tinlet = mat
            Tsol = Xsol[:,node]
            psol = rho*8.314*Tsol/M
            shape = (len(Xt),len(node))
            saveTransientSolution(path,Xt,Tsol,shape,name=name+'_T.txt')
            saveTransientSolution(path,Xt,psol,shape,name=name+'_p.txt')

        if  mat[0] == 'solid0D' or mat[0] == "cylindric0D" :
            Tsol = 0.5*(Xsol[:,node[0]]+Xsol[:,node[1]])
            shape = (len(Xt),1)
            saveTransientSolution(path,Xt,Tsol,shape,name=name+'_T.txt')


def saveMaterialSolutionSteady(path,matNodeList,matList,matNames,Xsol):

    for name,node,mat in zip(matNames,matNodeList,matList) :
        if  mat[0] == 'solid1D' or mat[0] == "cylindric1D" or mat[0]=="fluidCavity" or mat[0]=='PCM':
            Tsol = Xsol[node]
            saveSteadySolution(path,Tsol,name=name+'_T.txt')


        if mat[0] == 'gasCavity' :
            _,rho,cp,v,s,h,qint,M,qm,Tinlet = mat
            Tsol = Xsol[node]
            psol = rho*8.314*Tsol/M
            saveSteadySolution(path,Tsol,name=name+'_T.txt')
            saveSteadySolution(path,psol,name=name+'_p.txt')

        if  mat[0] == 'solid0D' or mat[0] == "cylindric0D" :
            Tsol = 0.5*(Xsol[node[0]]+Xsol[node[1]])
            saveSteadySolution(path,Tsol,name=name+'_T.txt')

def saveFinalSolution(path,Tsol):

    header = '-'*50+'\n'
    header += '     SOLUTION'+'\n'
    header += '\n'
    header += '-'*50+'\n'
    header += '\n'
    header += '|node 0|...|node n'+'|\n'

    np.savetxt(path+'/solution/final.txt',np.array([Tsol]),header=header,fmt='%.5e')


def buildMesh(path,matList,matNames):

    meshFile = open(path+"/solution/mesh.txt","w")
    file = ['#'+'-'*50]
    file += ['#']
    file += ['#     MESH']
    file += ['#']
    file += ['#'+'-'*50]
    file += ['# _____________________']
    file += ['# | node id |x pos[m] |']
    file += ['# |_________|_________|']
    file += ['']*2

    start = 0
    matNodeList = []
    matNodePos = []
    for klayer,(mat,name) in enumerate(zip(matList,matNames)) :

        if  mat[0] == 'solid1D' or mat[0] == "solid0D":
            _,k,rho,cp,np,e,qint = mat
            dx = e/np
            file += [name]
            file += ["{"]
            file += ["%d     %.5e"%(start+i,i*dx) for i in range(np+1)]
            file += ['}']
            file += ['']
            matNodeList.append([start+i for i in range(np+1)])
            matNodePos.append([i*dx for i in range(np+1)])
            start += np



        if mat[0] == "cylindric1D" or mat[0] == "cylindric0D" :
            _,k,rho,cp,np,r1,r2,qint = mat
            ri,re = min(r1,r2),max(r1,r2)
            e = re-ri
            dx = e/np
            file += [name]
            file += ["{"]
            file += ["%d     %.5e"%(start+i,ri + i*dx) for i in range(np+1)]
            file += ['}']
            file += ['']
            matNodeList.append([start+i for i in range(np+1)])
            matNodePos.append([ri+i*dx for i in range(np+1)])
            start += np


        if mat[0] == 'fluidCavity' :
            _,rho,cp,v,s,h,qint,qm,Tinlet = mat
            if klayer == 0 :
                file += [name]
                file += ["{"]
                file += ["%d     none"%(start)]
                file += ['}']
                file += ['']
                matNodeList.append([start])
                matNodePos.append([0])
                start += 1
            else :
                file += [name]
                file += ["{"]
                file += ["%d     none"%(start+1)]
                file += ['}']
                file += ['']
                matNodeList.append([start+1])
                matNodePos.append([0])
                start += 2

        if mat[0] == 'gasCavity' :
            _,rho,cp,v,s,h,qint,M,qm,Tinlet = mat
            if klayer == 0 :
                file += [name]
                file += ["{"]
                file += ["%d     none"%(start)]
                file += ['}']
                file += ['']
                matNodeList.append([start])
                matNodePos.append([0])
                start += 1
            else :
                file += [name]
                file += ["{"]
                file += ["%d     none"%(start+1)]
                file += ['}']
                file += ['']
                matNodeList.append([start+1])
                matNodePos.append([0])
                start += 2


        if mat[0] == 'PCM' :
            _,k ,rho,cp,np,qint,geom = mat
            if geom[0] == 'cartesian':
                e = geom[1]
                dx = e/np
                file += [name]
                file += ["{"]
                file += ["%d     %.5e"%(start+i,i*dx) for i in range(np+1)]
                file += ['}']
                file += ['']
                matNodeList.append([start+i for i in range(np+1)])
                matNodePos.append([i*dx for i in range(np+1)])
                start += np

            if geom[0] == 'cylindric':
                r1,r2 = geom[1],geom[2]
                ri,re = min(r1,r2),max(r1,r2)
                e = re-ri
                dx = e/np
                file += [name]
                file += ["{"]
                file += ["%d     %.5e"%(start+i,ri + i*dx) for i in range(np+1)]
                file += ['}']
                file += ['']
                matNodeList.append([start+i for i in range(np+1)])
                matNodePos.append([ri+i*dx for i in range(np+1)])
                start += np



    for line in file :
        meshFile.write(line+'\n')
    meshFile.close()

    return matNodeList,matNodePos


def plotTransient(matNames,matList,matNodes,Xt,Ysol,path) :

    # logo = plt.imread(os.sep.join(os.getcwd().split(os.sep)[:-2])+'/PythonScripts/Logo.png')

    #Evolutions Générales
    fig = plt.figure(0,figsize=(7.00,5.25),dpi=100)
    fig.clf()
    ax = fig.add_subplot(111)
    ax.set_prop_cycle(cycler('color',['#1f77b4', '#2ca02c', '#d62728', 'c',
                                      '#9467bd','orange', 'y','crimson',
                                      'darkseagreen','royalblue','coral']))
    Lines = []
    N0 = 0
    for name,nodes,mat in zip(matNames,matNodes,matList):

        if  mat[0] == 'solid1D' or mat[0] == "cylindric1D" or  mat[0]=='PCM':
            lw = 1.0
            line = ax.plot(Xt,Ysol[:,nodes[0]],lw=lw)
            Lines.append(line[0])
            ax.plot(Xt,Ysol[:,nodes[1:]],color=line[0].get_color(),lw=lw)


        if  mat[0] == 'gasCavity' or mat[0] == 'fluidCavity' :
            lw = 1.2
            line = ax.plot(Xt,Ysol[:,nodes],lw=lw)
            Lines.append(line[0])



        if  mat[0] == 'solid0D' or mat[0] == "cylindric0D" :
            lw = 1.0
            Tsol = (Ysol[:,nodes[0]]+Ysol[:,nodes[1]])/2
            line = ax.plot(Xt,Tsol,lw=lw)
            Lines.append(line[0])

    #plt.figimage(logo, 10, 10, zorder=1)
    plt.xlabel("Time $s$")
    plt.ylabel("Temperature $K$")
    plt.title("Temperatures")
    plt.grid(True)
    plt.margins(0.1)
    plt.legend(Lines,matNames,loc=0)
    plt.tight_layout()
    fig.savefig(path+"/plots/temperatures_plot.png")
    fig.clf()


    #Materiaux par Materiaux

    Lines = []
    N0 = 0
    fig = plt.figure(0,figsize=(7.00,5.25),dpi=100)
    fig.clf()
    for name,nodes,mat in zip(matNames,matNodes,matList):


        if  mat[0] == 'solid1D' :
            lw = 1.0
            _,_,_,_,np,e,_ = mat
            dx = e/np
            xVec = [i*dx for i in range(np+1)]

            ax1 = fig.add_subplot(211)
            ax1.plot(Xt,Ysol[:,nodes],lw=lw,color='b')
            ax1.set_xlabel("Time $s$")
            ax1.set_ylabel("Temperature $K$")
            ax1.grid(True)
            ax1.set_title(name)

            ax2 = fig.add_subplot(212)
            ax2.plot(xVec,Ysol[:,nodes].T,lw=lw,color='r')
            ax2.set_xlabel("x axis $m$")
            ax2.set_ylabel("Temperature $K$")
            ax2.grid(True)

        if  mat[0] == 'cylindric1D' :
            lw = 1.0
            _,_,_,_,np,r1,r2,_ = mat
            ri,re = min(r1,r2),max(r1,r2)
            e = re-ri
            dx = e/np
            xVec = [ri+i*dx for i in range(np+1)]

            ax1 = fig.add_subplot(211)
            ax1.plot(Xt,Ysol[:,nodes],lw=lw,color='b')
            ax1.set_xlabel("Time $s$")
            ax1.set_ylabel("Temperature $K$")
            ax1.grid(True)
            ax1.set_title(name)

            ax2 = fig.add_subplot(212)
            ax2.plot(xVec,Ysol[:,nodes].T,lw=lw,color='r')
            ax2.set_xlabel("r axis $m$")
            ax2.set_ylabel("Temperature $K$")
            ax2.grid(True)

        if mat[0] == 'PCM' :
            _,_ ,_,_,np,_,geom = mat
            if geom[0] == 'cartesian':
                lw = 1.0
                e = geom[1]
                dx = e/np
                xVec = [i*dx for i in range(np+1)]

                ax1 = fig.add_subplot(211)
                ax1.plot(Xt,Ysol[:,nodes],lw=lw,color='b')
                ax1.set_xlabel("Time $s$")
                ax1.set_ylabel("Temperature $K$")
                ax1.grid(True)
                ax1.set_title(name)

                ax2 = fig.add_subplot(212)
                ax2.plot(xVec,Ysol[:,nodes].T,lw=lw,color='r')
                ax2.set_xlabel("x axis $m$")
                ax2.set_ylabel("Temperature $K$")
                ax2.grid(True)

            if geom[0] == 'cylindric' :
                lw = 1.0
                r1,r2 = geom[1],geom[2]
                ri,re = min(r1,r2),max(r1,r2)
                e = re-ri
                dx = e/np
                xVec = [ri+i*dx for i in range(np+1)]

                ax1 = fig.add_subplot(211)
                ax1.plot(Xt,Ysol[:,nodes],lw=lw,color='b')
                ax1.set_xlabel("Time $s$")
                ax1.set_ylabel("Temperature $K$")
                ax1.grid(True)
                ax1.set_title(name)

                ax2 = fig.add_subplot(212)
                ax2.plot(xVec,Ysol[:,nodes].T,lw=lw,color='r')
                ax2.set_xlabel("r axis $m$")
                ax2.set_ylabel("Temperature $K$")
                ax2.grid(True)


        if  mat[0] == 'solid0D' or mat[0] == 'cylindric0D':
            lw = 1.2

            ax = fig.add_subplot(111)
            ax.plot(Xt,Ysol[:,nodes],lw=lw,color='b',ls='--')
            Tsol = (Ysol[:,nodes[0]]+Ysol[:,nodes[1]])/2
            ax.plot(Xt,Tsol,lw=lw,color='b',label='mean temp.')
            ax.set_xlabel("Time $s$")
            ax.set_ylabel("Temperature $K$")
            ax.grid(True)
            ax.set_title(name)
            ax.legend(loc=0)

        if  mat[0] == 'fluidCavity' :
            lw = 1.2

            ax = fig.add_subplot(111)
            ax.plot(Xt,Ysol[:,nodes],lw=lw,color='b')
            ax.set_xlabel("Time $s$")
            ax.set_ylabel("Temperature $K$")
            ax.grid(True)
            ax.set_title(name)

        if  mat[0] == 'gasCavity' :
            lw = 1.2
            _,rho,_,_,_,_,_,M,_,_ = mat
            Tsol = Ysol[:,nodes]
            psol = rho*8.314*Tsol/M

            ax1 = fig.add_subplot(211)
            ax1.plot(Xt,Tsol,lw=lw,color='b')
            ax1.set_xlabel("Time $s$")
            ax1.set_ylabel("Temperature $K$")
            ax1.grid(True)
            ax1.set_title(name)

            ax2 = fig.add_subplot(212)
            ax2.plot(Xt,psol,lw=lw,color='r')
            ax2.set_xlabel("Time $s$")
            ax2.set_ylabel("Pressure $Pa$")
            ax2.grid(True)
            ax2.set_title("pressure")

        plt.tight_layout()
        fig.savefig(path+"/plots/"+name+"_plot.png")
        fig.clf()



def plotSteady(matNames,matList,matNodes,Ysol,path) :

    # logo = plt.imread(os.sep.join(os.getcwd().split(os.sep)[:-2])+'/PythonScripts/Logo.png')

    #Evolutions Générales
    fig = plt.figure(0,figsize=(7.00,5.25),dpi=100)
    fig.clf()
    ax = fig.add_subplot(111)
    ax.set_prop_cycle(cycler('color',['#1f77b4', '#2ca02c', '#d62728', 'c',
                                      '#9467bd','orange', 'y','crimson',
                                      'darkseagreen','royalblue','coral']))
    Lines = []
    N0 = 0
    for name,nodes,mat in zip(matNames,matNodes,matList):

        lw = 1.2
        if (mat[0] == 'gasCavity') or (mat[0] == 'fluidCavity') :
            line = ax.plot(nodes,Ysol[nodes],'s')
        else :
            line = ax.plot(nodes,Ysol[nodes],lw=lw)
        Lines.append(line[0])

    #plt.figimage(logo, 10, 10, zorder=1)
    plt.xlabel("node index")
    plt.ylabel("Temperature $K$")
    plt.title("Temperatures")
    plt.grid(True)
    plt.margins(0.1)
    plt.legend(Lines,matNames,loc=0)
    plt.tight_layout()
    fig.savefig(path+"/plots/temperatures_plot.png")
    fig.clf()


    #Materiaux par Materiaux

    Lines = []
    N0 = 0
    fig = plt.figure(0,figsize=(7.00,5.25),dpi=100)
    fig.clf()
    for name,nodes,mat in zip(matNames,matNodes,matList):

        savePlotFlag = False
        if  mat[0] == 'solid1D' :
            lw = 1.2
            _,_,_,_,np,e,_ = mat
            dx = e/np
            xVec = [i*dx for i in range(np+1)]

            ax = fig.add_subplot(111)
            ax.plot(xVec,Ysol[nodes],lw=lw,color='b',marker='s',
                                    markerfacecolor='orange',
                                    markeredgecolor='b')
            ax.set_xlabel("x axis $m$")
            ax.set_ylabel("Temperature $K$")
            ax.grid(True)
            ax.set_title(name)

            savePlotFlag = True

        if  mat[0] == 'cylindric1D' :
            lw = 1.2
            _,_,_,_,np,r1,r2,_ = mat
            ri,re = min(r1,r2),max(r1,r2)
            e = re-ri
            dx = e/np
            xVec = [ri+i*dx for i in range(np+1)]

            ax = fig.add_subplot(111)
            ax.plot(xVec,Ysol[nodes].T,lw=lw,color='b',marker='s',
                                    markerfacecolor='orange',
                                    markeredgecolor='b')
            ax.set_xlabel("r axis $m$")
            ax.set_ylabel("Temperature $K$")
            ax.grid(True)
            ax.set_title(name)

            savePlotFlag = True

        if mat[0] == 'PCM' :
            _,_ ,_,_,np,_,geom = mat
            if geom[0] == 'cartesian':
                lw = 1.2
                e = geom[1]
                dx = e/np
                xVec = [i*dx for i in range(np+1)]

                ax = fig.add_subplot(111)
                ax.plot(xVec,Ysol[nodes].T,lw=lw,color='b',marker='s',
                                    markerfacecolor='orange',
                                    markeredgecolor='b')
                ax.set_xlabel("x axis $m$")
                ax.set_ylabel("Temperature $K$")
                ax.grid(True)
                ax.set_title(name)

            if geom[0] == 'cylindric' :
                lw = 1.2
                r1,r2 = geom[1],geom[2]
                ri,re = min(r1,r2),max(r1,r2)
                e = re-ri
                dx = e/np
                xVec = [ri+i*dx for i in range(np+1)]

                ax = fig.add_subplot(111)
                ax.plot(xVec,Ysol[nodes].T,lw=lw,color='b',marker='s',
                                    markerfacecolor='orange',
                                    markeredgecolor='b')
                ax.set_xlabel("r axis $m$")
                ax.set_ylabel("Temperature $K$")
                ax.grid(True)
                ax.set_title(name)

            savePlotFlag = True

        if savePlotFlag :
            plt.tight_layout()
            fig.savefig(path+"/plots/"+name+"_plot.png")
        fig.clf()