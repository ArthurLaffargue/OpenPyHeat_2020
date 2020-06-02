import os


def cleanCase(path) :
    print('-'*50)
    print('CLEANING SOLUTIONS AND PLOTS')
    listdir = os.listdir(path)
    if "solution" in listdir :
        for f in os.listdir(path+'/solution'):
            if f != 'initial.txt' :

                try : os.remove(path+'/solution/'+f)
                except : pass

    else :
        os.mkdir(path+'/solution')


    if "plots" in listdir :
        for f in os.listdir(path+'/plots'):
            try : os.remove(path+'/plots/'+f)
            except : pass

    else :
        os.mkdir(path+'/plots')
