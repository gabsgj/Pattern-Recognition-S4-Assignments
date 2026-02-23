import numpy as np # type: ignore
import math
import random


def init_matrix(Q):
    
    #initializing Pi and normalizing it
    lmbda=[[],[],[]]
    for i in range(Q):
        lmbda[0].append(random.random())
    lmbda[0]=np.array(lmbda[0])
    lmbda[0] = lmbda[0] / lmbda[0].sum()
    
    #initializing A and normalizing it
    for i in range(Q):
        lmbda[1].append([])
        for j in range(Q):
            lmbda[1][i].append(random.random())
    lmbda[1]=np.array(lmbda[1])
    lmbda[1] = lmbda[1] / lmbda[1].sum( axis=1, keepdims=True)
    
    #initializing B and normalizing it
    for i in range(Q):
        lmbda[2].append([])
        for j in range(2):
            lmbda[2][i].append(random.random())
    lmbda[2]=np.array(lmbda[2])
    lmbda[2] = lmbda[2] / lmbda[2].sum( axis=1, keepdims=True)
    return lmbda


def probability(O,a,b,pi):
    
    alpha=np.zeros((len(O),len(pi)))
    # alpha at time = 1 at each state = initial state probability(PI) x emission probability of the first observation (B)
    for i in range(len(pi)):
        alpha[0][i] = pi[i] * b[i][int(O[0])]
        
    # alpha at time t at each state = sum of (alpha at time t-1 at each state x transition probability from t-1th state to the t th state) x emission probability of the observation at time t    
    for t in range(1,len(O)):
        for j in range(len(pi)):
            sum1=0
            for i in range(len(pi)):
                sum1 += alpha[t-1][i] * a[i][j]
            alpha[t][j] = sum1 * b[j][int(O[t])]
    return sum(alpha[len(O)-1])


def forward(O,a,b,pi):
    
    alpha=np.zeros((len(O),len(pi)))
    # alpha at time = 1 at each state = initial state probability(PI) x emission probability of the first observation (B)
    for i in range(len(pi)):
        alpha[0][i] = pi[i] * b[i][int(O[0])]
        
    # alpha at time t at each state = sum of (alpha at time t-1 at each state x transition probability from t-1th state to the t th state) x emission probability of the observation at time t    
    for t in range(1,len(O)):
        for j in range(len(pi)):
            sum1=0
            for i in range(len(pi)):
                sum1 += alpha[t-1][i] * a[i][j]
            alpha[t][j] = sum1 * b[j][int(O[t])]
    return alpha

def backward(O,a,b,pi):
    
    beta=np.zeros((len(O),len(pi)))
    # beta at time = T at each state = 1
    for i in range(len(pi)):
        beta[len(O)-1][i] = 1
        
    # beta at time t at each state = sum of (beta at time t+1 at each state x transition probability from t th state to the t+1 th state) x emission probability of the observation at time t+1    
    for t in range(len(O)-2,-1,-1):
        for i in range(len(pi)):
            sum1=0
            for j in range(len(pi)):
                sum1 += a[i][j] * beta[t+1][j] * b[j][int(O[t+1])]
            beta[t][i] = sum1
    return beta

def modify_lmbda(O,lmbda):
    
    a=lmbda[1]
    b=lmbda[2]
    pi=lmbda[0]
    
    gamma=np.zeros((len(O),len(pi)))
    alpha=forward(O,a,b,pi)
    beta=backward(O,a,b,pi)
    #finding gamma probabilty
    for t in range(len(O)):
        for i in range(len(pi)):
            gamma[t][i] = (alpha[t][i] * beta[t][i]) /probability(O,a,b,pi)
            
    #modified PI
    for i in range(len(pi)):
        pi[i] = gamma[0][i]
    
    # eta probability of being in state i at time t and state j at time t+1 given the observation sequence O and the model parameters lmbda
    eta=np.zeros((len(O)-1,len(pi),len(pi)))
    for t in range(len(O)-1):
        for i in range(len(pi)):
            for j in range(len(pi)):
                eta[t][i][j] = (alpha[t][i] * a[i][j] * b[j][int(O[t+1])] * beta[t+1][j]) / probability(O,a,b,pi)    
    
    #modified A
    for i in range(len(pi)):
        for j in range(len(pi)):
            sum1=sum(eta[t][i][j] for t in range(len(O)-1))
            sum2=sum(gamma[t][i] for t in range(len(O)-1))
            a[i][j] = sum1/sum2
    
    #modified B
    for i in range(len(pi)):
        for j in range(2):
            sum1=sum(gamma[t][i] for t in range(len(O)) if int(O[t])==j)
            sum2=sum(gamma[t][i] for t in range(len(O)))
            b[i][j] = sum1/sum2
            
    
    return [pi,a,b]

tolerance=0.0000001
O=input("Enter observation sequence: ")
Q=int(input("Enter no of states: "))
lmbda=init_matrix(Q)
pro_O_old=0
pro_O_new=probability(O,lmbda[1],lmbda[2],lmbda[0])


while (abs(pro_O_new - pro_O_old) > tolerance):
    pro_O_old=pro_O_new
    lmbda=modify_lmbda(O,lmbda)
    pro_O_new=probability(O,lmbda[1],lmbda[2],lmbda[0])
    print("Probability of the observation sequence: \n",pro_O_new)
    
    

print("Initial state probabilities: \n",lmbda[0])
print("Transition probabilities: \n",lmbda[1])
print("Emission probabilities: \n",lmbda[2])

print("Probability of the observation sequence: \n",pro_O_new)

