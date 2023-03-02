from hmm import *
def trainedparam(num):
    with open('m{number}.npy'.format(number=num), 'rb') as f:
        a = np.load(f)
        b = np.load(f)
        c = np.load(f)
        return a,b,c

init1,BMtrainedTransition1,BMtrainedEmission1=trainedparam(1)
init2,BMtrainedTransition2,BMtrainedEmission2=trainedparam(2)
init3,BMtrainedTransition3,BMtrainedEmission3=trainedparam(3)
init4,BMtrainedTransition4,BMtrainedEmission4=trainedparam(4)
init5,BMtrainedTransition5,BMtrainedEmission5=trainedparam(5)

def tstToInt(test):
    L=["A","B","C","D","E","F"]
    toint=test
    for i in range(6):
        toint=np.char.replace(toint,str(L[i]),str(i))
    count=0
    testToInt=[]
    for line in toint:
        for word in toint[count]:
            testToInt=np.append(testToInt,int(word))
        count+=1
    testToInt=testToInt.astype(int)
    testToInt=testToInt.reshape((2500,50))
    return  testToInt
"""""""""""""""""""""""""""
test1=np.loadtxt("testing_data1.txt",dtype=str)
test1=tstToInt(test1)
test2=np.loadtxt("testing_data2.txt",dtype=str)
test2=tstToInt(test2)
"""""""""""""""""""""""""""
def evalwithViterbi(testfile,num):
    file = open("result{}.txt".format(num), "w+")
    for i in range(testfile.shape[0]):
        prob_evaluate=np.zeros((5))
        s, p1 = viterbi(testfile[i], BMtrainedTransition1, BMtrainedEmission1, initial_dist)
        s, p2 = viterbi(testfile[i], BMtrainedTransition2, BMtrainedEmission2, initial_dist)
        s, p3 = viterbi(testfile[i], BMtrainedTransition3, BMtrainedEmission3, initial_dist)
        s, p4 = viterbi(testfile[i], BMtrainedTransition4, BMtrainedEmission4, initial_dist)
        s, p5 = viterbi(testfile[i], BMtrainedTransition5, BMtrainedEmission5, initial_dist)
        prob_evaluate=[p1,p2,p3,p4,p5]
        m=np.argmax(prob_evaluate)
        file.write("model_0{n}.txt {pe} \n ".format(n=m+1,pe=prob_evaluate[m]))
    file.close()
""""""""""""""""""""""    
evalwithViterbi(test1,1)
evalwithViterbi(test2,2)
"""""""""""""""""""""
###calculate accuracy
testing_answer =np.loadtxt("testing_answer.txt",dtype=str)
result1=np.loadtxt("result1.txt",dtype=str)
count =0
for i in range(result1.shape[0]):
    if(result1[i][0][7]!= testing_answer[i][7]):
        count+=1
print(count/2500)

if __name__ == "__main__":
    numb= input("Which trainig file?:")
    testf=np.loadtxt("testing_data{nn}.txt".format(nn=numb),dtype=str)
    testf = tstToInt(testf)
    evalwithViterbi(testf,numb)