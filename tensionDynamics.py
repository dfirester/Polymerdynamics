import numpy as np
import time
import scipy.integrate as integrate
import scipy.special as special
from multiprocessing import Pool

# Here we define the integrands to be used in the computation of the solution
def jp1(x,t,F_t,f0):
    return 2*x**2*(np.exp(-2*x**2*(x**2 *t + F_t))) / (x**2 + f0)

def jp2(x,t,ti,tim1,F_t,F_ti,F_tim1):
    return (2*x**2/(x**2 + (F_ti - F_tim1)/(ti - tim1))) * (np.exp(-2*x**2*(x**2*(t-ti) + F_t - F_ti)) - np.exp(-2*x**2*(x**2*(t-tim1) + F_t - F_tim1)))

def jp2_1(x,t,ti,tim1,F_t,F_ti,F_tim1):
    return (1/(x**2 + (F_ti - F_tim1)/(ti - tim1))**2)*(1/(ti - tim1))*(1 - np.exp(-2*x**2*(x**2*(ti-tim1) + F_ti - F_tim1))) + ((2*x**2)/(x**2 + (F_ti - F_tim1)/(ti - tim1)))*np.exp(-2*x**2*(x**2*(ti-tim1) + F_ti - F_tim1))

def p1(x,t,F_t,f0):
    return (1-np.exp(-2*x**2*(x**2 * t + F_t)))/(x**2 + f0)

def p2(x,t,ti,tim1,F_t,F_ti,F_tim1):
    return -(1/(x**2 + (F_ti - F_tim1)/(ti - tim1))) * (np.exp(-2*x**2*(x**2*(t-ti) + F_t - F_ti)) - np.exp(-2*x**2*(x**2*(t-tim1) + F_t - F_tim1)))


def integral(args):
    k = args[0]
    j = args[1]
    dt = args[2]
    F_test = args[3]
    F_target_k_1 = args[4]
    F_target_k = args[5]
    I = integrate.quad(p2,0,np.inf,args=(j*dt,(k+1)*dt,k*dt,F_test,F_target_k_1,F_target_k))
    return I[0]

def integralJ(args):
    k = args[0]
    j = args[1]
    dt = args[2]
    F_test = args[3]
    F_target_k_1 = args[4]
    F_target_k = args[5]
    if k + 1 == j:
        I = integrate.quad(jp2_1,0,np.inf,args=(j*dt,(k+1)*dt,k*dt,F_test,F_target_k_1,F_target_k))
    else:
        I = integrate.quad(jp2,0,np.inf,args=(j*dt,(k+1)*dt,k*dt,F_test,F_target_k_1,F_target_k))
    return I[0]


class Polymer(object):
    '''Creates a polymer object to simulate. The specific polymer has an associated persistant length and 
    contour length, along with relevent boundry conditions (pulling protocols) that govern its behavior '''
    # Class Attributes
    lcl_dt = None
    lcl_dx = None
    N = None
    f0 = None
    ten_prof = []
    F_prof = []
    boundry = []

    
    # Initalizer / Instance Attributes
    def __init__(self,lp,lc):
        '''Construct a Polymer object.
            Args:
                lp: The persistance length of the chain
                lc: The contour length of the chain '''
        self.lp = lp
        self.lc = lc
    
    # Allow user to set the time step of the simulation
    def setdt(self,dt):
        self.lcl_dt = dt

    # Allow user to set the number of points to break the simulation into along the arclength coordinate
    def setNx(self,N):
        self.lcl_dx = self.lc/N
        self.N = N


    def boundry(self,sympull):
        self.boundry = sympull[:]

    
    def setValues(self,dt,N,f0,sympull):
        self.lcl_dx = self.lc/N
        self.N = N
        self.lcl_dt = dt
        self.f0 = f0
        self.boundry = sympull[:]

    @property
    def J1m(self):
        matrix = np.zeros((self.N,self.N))
        for i in range(0,self.N):
            matrix[i,i] = -2
            if i-1 > -1:
                matrix[i,i-1] = 1
            if i+1 < self.N:
                matrix[i,i+1] = 1
           # matrix[i,i] = -30
            #if i-1 > -1:
             #   matrix[i,i-1] = 16
           # if i+1 < 100:
            #    matrix[i,i+1] = 16
            #if i-2 > -1:
            #    matrix[i,i-2] = -1
            #if i+2 < 100:
            #    matrix[i,i+2] = -1
        return matrix

  

    

        


    # Feed in the boundry conditions
    
    
    def Simulation(self,T,numthreads):
        numT = int(np.floor(T/self.lcl_dt))
        dt = self.lcl_dt
        J2m = np.zeros((self.N,self.N))
        self.ten_prof = np.zeros((self.N,numT))
        f_test = np.zeros(self.N)
        self.F_prof = np.zeros((self.N,numT))
        F_test = np.zeros(self.N)
        error = np.zeros(self.N)
        P = np.zeros(self.N)
        P1 = np.zeros(self.N)
        P2 = np.zeros(self.N)
        P1J = np.zeros(self.N)
        P2J = np.zeros(self.N)
        tosubtract = np.zeros(self.N)
        self.ten_prof[:,0] = self.f0
        self.F_prof[:,0] = 0
        # INITATE THE TEST VALUE TO START
        f_test = np.linspace(self.boundry[1],self.boundry[1],self.N)

        p = Pool(numthreads)

        starttimes = np.zeros(numT)
        starttimes[0] = time.time()
        # j indexes the time axis. We compute the target f_target for each time point, then move on
        for j in range(1,numT):
            print('time elapsed:',time.time() - starttimes[j-1])
            starttimes[j] = time.time()
    
            print(j)
            for trial in range(0,200):
            
                # CALCULATE THE VALUE OF THE TEST INTEGRAL FOR THE GIVEN TEST FUNCTION AND PREVIOUS ANSWERS
                F_test[:] = 0
                F_test[:] = self.F_prof[:,j-1]
                F_test[:] = F_test[:] + dt*(f_test[:] + self.ten_prof[:,j-1])/2
        
                self.F_prof[:,j] = F_test[:]
        
                #CALCULATE THE RHS OF THE EQUATION: THE INVERSE TRANSFORM FROM WAVE NUMBER TO S
                #EACH SPATIAL POINT IS INDEPENDENT, SO WE CAN CALCULATE EACH ONE IN TURN

                
                for i in range(0,self.N):
            
                    #Calculate part 1 of the RHS of equation, a simple integral
                    I = integrate.quad(p1,0,np.inf,args=(j*dt,F_test[i],self.f0))
                    P1[i] = I[0]
            
                    E = integrate.quad(jp1,0,np.inf,args=(j*dt,F_test[i],self.f0))
                    P1J[i] = E[0]

                    parameters = []
                    for z in range(0,j):
                        k = j - z - 1
                        parameters.extend([[k,j,dt,F_test[i],self.F_prof[i,k+1],self.F_prof[i,k]]])
                    results_pooled = list(p.map(integral,parameters))
                    results = list(results_pooled)
                    P2[i] = sum(results)
            
                    results_pooled_J = list(p.map(integralJ,parameters))
                    results_J = list(results_pooled_J)
                    P2J[i] = sum(results_J)
            
                    #Calculate part 2 of the RHS of the equation for each value of tp. This creates a vector of
                    #values that must be summed.
                    #vector = np.zeros(j) #This vector holds the approximate values of the integral for each tp
                    #partialsum = np.zeros(j+1) #This vector will hold the partial sums as we converge on the integral
                                     #When this is not changing much, we will truncate the integral in order to
                                     #Save computation time
                    

                    #for z in range(0,j):
                     #   k = j - z - 1
                     #   I = integrate.quad(p2,0,np.inf,args=(j*dt,(k+1)*dt,k*dt,F_test[i],self.F_prof[i,k+1],self.F_prof[i,k]))
                     #   vector[k] = I[0]
                     #   partialsum[z+1] = sum(vector)
                
           
                
                     # if (abs(partialsum[z+1] - partialsum[z])) < .1:
                     #    count = count + 1
                    
                   # if count == 3:
                   #     print('break')
                    #    break
               
                
                   # P2[i] = sum(vector)
                
                ## THIS IS FOR COMPUTING THE JACOBIAN    
                  #  vectorJ = np.zeros(j) #This vector holds the approximate values of the integral for each tp
                  #  partialsumJ = np.zeros(j+1) #This vector will hold the partial sums as we converge on the integral
                                         #When this is not changing much, we will truncate the integral in order to
                                         #Save computation time
            
                 #   for z in range(0,j):
                 #       k = j - z - 1
                  #      if z == 0:
                   #         I = integrate.quad(jp2_1,0,np.inf,args=(j*dt,(k+1)*dt,k*dt,F_test[i],self.F_prof[i,k+1],self.F_prof[i,k]))
                   #     else:
                  #          I = integrate.quad(jp2,0,np.inf,args=(j*dt,(k+1)*dt,k*dt,F_test[i],self.F_prof[i,k+1],self.F_prof[i,k]))
                  #      vectorJ[k] = I[0]
                    #    partialsumJ[z+1] = sum(vectorJ)
                
                
    
                   # P2J[i] = vectorJ[j-1]
            

                # Sum the two contributions
                P = P1 + P2
        
                

                for i in range(0,self.N-1):
                    J2m[i,i] = (P1J[i] + P2J[i])*(1/self.lp)
            
                Jacobian = self.J1m - J2m
        
   
    
        
                #TAKE 2ND DERIVATIVE OF THE TEST FUNCTION, TO COMPUTE CURVATURE
                dxdy = np.gradient(F_test)
                curvature = np.gradient(dxdy)
                for i in range(0,self.N):
                    if np.abs(curvature[i]) > 1e3:
                        curvature[i] = curvature[i-1]

                curvature = curvature
                error = curvature - (1/self.lp)*P
        
                error[0] = 0
                error[self.N-1] = 0
              
        

        #for i in range(0,1000):
         #   intval[i] = integrate.trapz(curvature[0:i])
         #   S[i] = integrate.trapz(intval[0:i])

       # C = 1 - S[0]
       # B = (1 - C - S[999])/999
   
                if ((np.sum(abs(error)))) < .001:
                    self.ten_prof[:,j] = f_test
                    self.F_prof[:,j] = F_test
                    print('success')
                    print('Num Trials:',trial)
                    np.savetxt('ten_vals.txt',self.ten_prof)
                    np.savetxt('f_vals.txt',self.F_prof)
                    break #Move on to next time step
    
        
        
                tosubtract = (np.dot(np.linalg.pinv(Jacobian),error))
                F_test = F_test - tosubtract
        
                #f_test = f_test - tosubtract
                f_test = (2/dt)*(F_test - self.F_prof[:,j-1]) - self.ten_prof[:,j-1]
        
        
                f_test[0] = self.boundry[j]
                f_test[self.N-1] = self.boundry[j]

                
        p.terminate()
 