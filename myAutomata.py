import numpy as np
import pandas as pd
from scipy import signal
import warnings

#suppress warnings
warnings.filterwarnings('ignore')

class automata:
    def __init__(self,w,h,rules,boundary,state_matrix):
        self.w = w
        self.h = h
        self.rules = rules
        self.boundary = boundary
        self.state_matrix = state_matrix
        self.N = self.w*self.h

    def evolve_system(self,bdry):
        counter = 0
        B = eval(self.rules[0])
        D = eval(self.rules[1])
        if bdry=='closed':
            newMatrix = np.zeros((self.w,self.h),dtype=int)
            kernel = np.ones((3,3),dtype=np.int8)
            kernel[1,1] = 0
            k = signal.convolve(self.state_matrix,kernel,mode='same')

            for i in range(self.w):
                for j in range(self.h):
                    n = k[i,j]

                    if n in B:
                        if newMatrix[i][j] != 2:
                            newMatrix[i][j] += 1

                    elif n in D:
                        if newMatrix[i][j] != 0:
                            newMatrix[i][j] -= 1
                    else:
                        newMatrix[i][j] = self.state_matrix[i][j]
                        counter+=1
            old_matrix = self.state_matrix
            self.state_matrix = newMatrix
            return [old_matrix,newMatrix,counter]
        

        elif bdry=='periodic':
            newMatrix = np.zeros((self.w,self.h),dtype=int)
            kernel = np.ones((3,3),dtype=np.int8)
            kernel[1,1] = 0
            k = signal.convolve(self.state_matrix,kernel,mode='same')
            k[0][0] += self.state_matrix[99][99] +  self.state_matrix[99][0] +  self.state_matrix[99][1] +  self.state_matrix[0][99] + self.state_matrix[1][99]
            k[99][0] += self.state_matrix[0][0] +  self.state_matrix[1][0] +  self.state_matrix[99][0] +  self.state_matrix[99][99] + self.state_matrix[99][98]
            k[0][99] += self.state_matrix[0][0] +  self.state_matrix[0][1] +  self.state_matrix[99][0] +  self.state_matrix[99][99] + self.state_matrix[98][99]
            k[99][99] += self.state_matrix[0][99] +  self.state_matrix[0][98] +  self.state_matrix[0][0] +  self.state_matrix[99][0] + self.state_matrix[99][98]

            for i in range(1,99):
                k[i][0] += self.state_matrix[i-1][99] + self.state_matrix[i][99] + self.state_matrix[i+1][99] 
                k[i][99] += self.state_matrix[i-1][0] + self.state_matrix[i][0] + self.state_matrix[i+1][0]
                k[0][i] += self.state_matrix[99][i-1] + self.state_matrix[99][i] + self.state_matrix[99][i+1] 
                k[99][i] += self.state_matrix[0][i-1] + self.state_matrix[0][i] + self.state_matrix[0][i+1]

            for i in range(self.w):
                for j in range(self.h):

                    n = k[i,j]

                    if n in B:
                        if newMatrix[i][j] != 2:
                            newMatrix[i][j] += 1

                    elif n in D:
                        if newMatrix[i][j] != 0:
                            newMatrix[i][j] -= 1
                         
                    else:
                        newMatrix[i][j] = self.state_matrix[i][j]
                        counter+=1
            old_matrix = self.state_matrix
            self.state_matrix = newMatrix
            return [old_matrix,newMatrix,counter]


    def to_benchmark(self):
        curr_mat = self.state_matrix

        n1 = np.count_nonzero(curr_mat==2)
        n2 = np.count_nonzero(curr_mat==1)

        E = n1 + 2*n2

        if( n1 != 0 or n2 !=0 ):
            S = - (self.N-(n1+n2))*np.log((self.N -n1 -n2)/self.N) - n1*np.log(n1/self.N) - n2*np.log(n2/self.N)
            T = -(1/(np.log(n1/self.N) + 1 ) + 2/(np.log(n2/self.N) +1))
            C = 1 / ( 1 / (n1* (np.log(n1/self.N) + 1) * (np.log(n1/self.N) + 1) )   ) +  4 / ( n2 * (np.log(n2/self.N) + 1) * (np.log(n2/self.N) + 1) )
            A = E - T*S
        else:
            S = 0
            T = 0
            C = np.inf
            A = np.inf

        E_bdry=0
        for i in range(100):
            E_bdry += curr_mat[0][i] + curr_mat[i][99] + curr_mat[i][0] + curr_mat[99][i]

        P = E_bdry/(4*np.sqrt(self.N) - 4)

        return [E,T,C,S,A,P]


    def to_partition(self,diff):
        T_a = diff/self.N
        if(T_a != 0):                
            Z = (np.exp(-2*self.N/T_a)*(np.exp((2*self.N+1)/T_a )-1)) /(np.exp(1/T_a) -1)
            E = 1 + np.exp(1/T_a)/(np.exp(1/T_a) -1)
            C = (E - 1)/(T_a*T_a)
            S = E /T_a + np.log(Z)
            A = -T_a*np.log(Z)
        #with limit values of course.
        elif (T_a == 0): 
            Z=1
            E=2
            C = np.inf
            S= np.inf
            A = 0

        return [E,T_a,C,S,A,Z]
    
    
    def write_to_class(self,class1,B,D,cls):
        class1.loc[len(class1.index)] = [f'B:{B} D: {D}',f'{cls}']
        class1.to_csv(f'{boundary}_{self.w}x{self.h}/benchmark.csv',index=False)


    
    def to_ideal(self):
        curr_mat = self.state_matrix

        n1 = np.count_nonzero(curr_mat==2)
        n2 = np.count_nonzero(curr_mat==1)

        E = n1 + 2*n2
        T_a = E/self.N
        C = self.N 
        S = self.N*np.log(E)
        A = E - E*np.log(E)
        P = T_a*np.log(E)

        return [E,T_a,C,S,A,P]
    




if __name__ == "__main__":
    
    ruleset = pd.read_csv('rulelist.csv')
    r = [['closed',400],['peridoic',400]]
    for k,j in r:
        w=j
        h=j
        boundary=k

        
        stateMatrix = np.random.uniform(0,1,(w,h))
        stateMatrix = (stateMatrix < 0.1).astype(int) # Initial state density of alive cells.

        A = np.nonzero(stateMatrix)

        for x in range(len(A[0])):
            stateMatrix[A[0][x]][A[1][x]] += int((np.random.uniform(0,1) < 0.5))

        del A

        # For storing benchmark, ideal gas and partition function values of E,T,C,S,A,P. i=0 is for benchmark, i=1 is for parittion function and i=2 is for ideal gas. i stands for rows of E[i][j]
       
        benchmark = pd.read_csv(f'{boundary}_{str(w)}x{str(h)}/benchmark.csv')
        partit = pd.read_csv(f'{boundary}_{str(w)}x{str(h)}/partition.csv')
        ideal = pd.read_csv(f'{boundary}_{str(w)}x{str(h)}/ideal_gas.csv')
        class1= pd.read_csv(f'{boundary}_{str(w)}x{str(h)}/classes.csv')
        for i in range(ruleset.__len__()):
            rows, cols = (3, 100)
            E = [[0]*cols]*rows
            T = [[0]*cols]*rows
            C = [[0]*cols]*rows
            S = [[0]*cols]*rows
            A = [[0]*cols]*rows
            P = [[0]*cols]*rows
            Z = [0]*cols # It is only in partition function approach.
            
            rules = (ruleset["B"][i],ruleset["D"][i])
            automata1 =  automata(w,h,rules,boundary,stateMatrix)

            B = rules[0]
            D = rules[1]
            for t in range(100):
                values = automata1.evolve_system(boundary)

                if(values[2]==w*h):
                    automata1.write_to_class(class1,B,D,'Class E')
                    break
                
                difference = np.sum(np.abs(values[1]-values[0]))      

                E[0][t],T[0][t],C[0][t],S[0][t],A[0][t],P[0][t] = automata1.to_benchmark()
                E[1][t],T[1][t],C[1][t],S[1][t],A[1][t],Z[t] = automata1.to_partition(difference)
                E[2][t],T[2][t],C[2][t],S[2][t],A[2][t],P[2][t] = automata1.to_ideal()
                
                
                print(i)
        
                       
            
            benchmark.loc[len(benchmark.index)] = [f'B:{B} D: {D}',E[0],T[0],C[0],S[0],A[0],P[0]]

            partit.loc[len(partit.index)]=[f'B:{B} D: {D}',E[1],T[1],C[1],S[1],A[1],Z]
            
            ideal.loc[len(ideal.index)]=[f'B:{B} D: {D}',E[2],T[2],C[2],S[2],A[2],P[2]]
            

        benchmark.to_csv(f'{boundary}_{str(w)}x{str(h)}/benchmark.csv',index=False)
        partit.to_csv(f'{boundary}_{str(w)}x{str(h)}/partition.csv',index=False)
        ideal.to_csv(f'{boundary}_{str(w)}x{str(h)}/ideal_gas.csv',index=False)
        
        
