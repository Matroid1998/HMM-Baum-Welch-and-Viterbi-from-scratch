import numpy as np
from hmm import *
obs=np.loadtxt("model_init.txt",skiprows=12,delimiter="\t")
transition = np.array([[0.3,0.3,0.1,0.1,0.1,0.1]
                       ,[0.1,0.3,0.3,0.1,0.1,0.1]
                       ,[0.1,0.1,0.3,0.3,0.1,0.1]
                       ,[0.1,0.1,0.1,0.3,0.3,0.1]
                       ,[0.1,0.1,0.1,0.1,0.3,0.3]
                       ,[0.3,0.1,0.1,0.1,0.1,0.3]])
initial_dist = np.array([0.2,0.1,0.2,0.2,0.2,0.1])
def inputToInt(train):
    L = ["A", "B", "C", "D", "E", "F"]
    toint = train
    for i in range(6):
        toint = np.char.replace(toint, str(L[i]), str(i))
    trained=toint.astype(int)
    return  trained
def readTrainData(num):
    f= open("seq_model_0{number}.txt".format(number=str(num)))
    trainData=f.read().replace("\n","")
    f.close()
    train=np.array(list(trainData))
    return  train
train1=readTrainData(1)
train1=inputToInt(train1)
train2=readTrainData(2)
train2=inputToInt(train2)
train3=readTrainData(3)
train3=inputToInt(train3)
train4=readTrainData(4)
train4=inputToInt(train4)
train5=readTrainData(5)
train5=inputToInt(train5)
### trainig:
""""""""""""""""""""""
BMtrainedTransition1,BMtrainedEmission1,init1 = baum_welch(train1,transition,obs.transpose(),initial_dist,n_iter=20)
BMtrainedTransition2,BMtrainedEmission2,init2 = baum_welch(train2,transition,obs.transpose(),initial_dist,n_iter=20)
BMtrainedTransition3,BMtrainedEmission3,init3 = baum_welch(train3,transition,obs.transpose(),initial_dist,n_iter=20)
BMtrainedTransition4,BMtrainedEmission4,init4 = baum_welch(train4,transition,obs.transpose(),initial_dist,n_iter=30)
BMtrainedTransition5,BMtrainedEmission5,init5 = baum_welch(train5,transition,obs.transpose(),initial_dist,n_iter=30)
"""""""""""""""""""""
###saving model parameters in npy file
"""""""""""""""""""""""""""""
with open('m1.npy', 'wb') as f:
    np.save(f,init1)
    np.save(f,BMtrainedTransition1)
    np.save(f,BMtrainedEmission1)
with open('m2.npy', 'wb') as f:
    np.save(f,init1)
    np.save(f,BMtrainedTransition2)
    np.save(f,BMtrainedEmission2)
with open('m3.npy', 'wb') as f:
    np.save(f,init3)
    np.save(f,BMtrainedTransition3)
    np.save(f,BMtrainedEmission3)
with open('m4.npy', 'wb') as f:
    np.save(f,init4)
    np.save(f,BMtrainedTransition4)
    np.save(f,BMtrainedEmission4)
with open('m5.npy', 'wb') as f:
    np.save(f,init5)
    np.save(f,BMtrainedTransition5)
    np.save(f,BMtrainedEmission5)
"""""""""""""""""""""""""""""""""""
###saving in text file


def saveArray(trans,emission,initial,numb):
    file = open("model_0{nu}.txt".format(nu=str(numb)), "w+")
    init_distr = str(initial).replace(']','')
    init_distr=init_distr.replace('[','')
    init_distr=init_distr.strip()
    file.write("initial_dist: \n ")
    file.write(init_distr)
    file.write("\n")
    content = str(trans).replace(']','')
    content=content.replace('[','')
    content=content.strip()
    file.write("transmition: \n ")
    file.write(content)
    file.write("\n")
    emis=str(emission).replace(']','')
    emis=emis.replace('[','')
    emis=emis.strip()
    file.write("emission: \n ")
    file.write(emis)
    file.close()
""""""""""""""""""""""""""""    
saveArray(BMtrainedTransition1,BMtrainedEmission1.T,init1,1)    
saveArray(BMtrainedTransition2,BMtrainedEmission2.T,init2,2)
saveArray(BMtrainedTransition3,BMtrainedEmission3.T,init3,3)
saveArray(BMtrainedTransition4,BMtrainedEmission4.T,init4,4)
saveArray(BMtrainedTransition5,BMtrainedEmission5.T,init5,5)
"""""""""""""""""""""""""""""""""
if __name__ == "__main__":
    n = int(input("number of iteration?"))
    model = int(input("which trainig set?"))
    train = readTrainData(model)
    train = inputToInt(train)
    BMtrainedTransition,BMtrainedEmission,init=baum_welch(train,transition,obs,initial_dist,n)
    saveArray(BMtrainedTransition,BMtrainedEmission,init,model)
