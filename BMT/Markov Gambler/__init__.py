#this is a program for assignment 10
import numpy as np

#initial value
k = 10
#goal value K
K = 20
#probability for red
p = 18/37
q = 1 - p
#transition matrix
A = np.zeros((K+1, K+1))

#define state vector
pi = np.zeros((1,K+1))

#set up initial state vector
def init_vector(k):
  for i in range (K):
    if(i==k):pi[0,i] = 1
    else: pi[0,i] = 0

#setup transition matrix
for i in range (1,K):
    for j in range(0,K+1):

        if(j == (i+1)): A[i,j] = 18/37
        elif(j == (i-1)): A[i,j] = 19/37

A[0,0] = 1;
A[K,K] = 1;

#main loop
#t is iteration times
init_vector(k)
for t in range(100000):
    pi = np.dot(pi,A)
print(pi)
