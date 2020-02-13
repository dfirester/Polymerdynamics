import numpy as np
import sys
import time
import scipy.integrate as integrate
import scipy.special as special
from multiprocessing import Pool

# Here we define the integrands to be used in the computation of the solution
def jp1(x,t,F_t,f0,k):
    return 2*x**2*(np.exp(-2*x**2*(x**2 *t + F_t))) / (x**2 + f0/k)

def jp2(x,t,ti,tim1,F_t,F_ti,F_tim1):
    return (2*x**2/(x**2 + (F_ti - F_tim1)/(ti - tim1))) * (np.exp(-2*x**2*(x**2*(t-ti) + F_t - F_ti)) - np.exp(-2*x**2*(x**2*(t-tim1) + F_t - F_tim1)))

def jp2_1(x,t,ti,tim1,F_t,F_ti,F_tim1):
    return (1/(x**2 + (F_ti - F_tim1)/(ti - tim1))**2)*(1/(ti - tim1))*(1 - np.exp(-2*x**2*(x**2*(ti-tim1) + F_ti - F_tim1))) + ((2*x**2)/(x**2 + (F_ti - F_tim1)/(ti - tim1)))*np.exp(-2*x**2*(x**2*(ti-tim1) + F_ti - F_tim1))

def p1(x,t,F_t,f0,k):
    return (1-np.exp(-2*x**2*(x**2 * t + F_t)))/(x**2 + f0/k)

def p2(x,t,ti,tim1,F_t,F_ti,F_tim1):
    return -(1/(x**2 + (F_ti - F_tim1)/(ti - tim1))) * (np.exp(-2*x**2*(x**2*(t-ti) + F_t - F_ti)) - np.exp(-2*x**2*(x**2*(t-tim1) + F_t - F_tim1)))


def integral(args):
    tn = args[0]
    ti = args[1]
    tim1 = args[2]
    F_test = args[3]
    F_target_k_1 = args[4]
    F_target_k = args[5]
    I = integrate.quad(p2,0,np.inf,args=(tn,ti,tim1,F_test,F_target_k_1,F_target_k))
    return I[0]

def integralJ(args):
    tn = args[0]
    ti = args[1]
    tim1 = args[2]
    F_test = args[3]
    F_target_k_1 = args[4]
    F_target_k = args[5]
    if tn == ti:
        I = integrate.quad(jp2_1,0,np.inf,args=(tn,ti,tim1,F_test,F_target_k_1,F_target_k))
    else:
        I = integrate.quad(jp2,0,np.inf,args=(tn,ti,tim1,F_test,F_target_k_1,F_target_k))
    return I[0]


class Polymer(object):
    '''Creates a polymer object to simulate. The specific polymer has an associated persistant length and 
    contour length, along with relevent boundry conditions (pulling protocols) that govern its behavior '''
    # Class Attributes
    lcl_dx = None
    N = None
    f0 = None
    real_dt = None
    numT = None
    ten_prof = []
    F_prof = []
    boundry = []
    error = []
    sld = []
    lcl_dt = []
    lp = []
    

    
    # Initalizer / Instance Attributes
    def __init__(self,lc,transdrag):
        '''Construct a Polymer object.
            Args:
                lp: The persistance length of the chain
                lc: The contour length of the chain '''
        self.lc = lc
        self.tdrag = transdrag
        self.ldrag = transdrag/2
    
    # Allow user to set the time step of the simulation
    def setdt(self,dt):
        ''' Set the time step to use in the simulation. This is the scaled time step, which ranges from
        .01 to 10-100, depending on the pulling force and speed. Also calculate the real time step in real time
        Args:
                dt: The rescaled timestep
                '''
        self.lcl_dt = dt
        self.real_dt = dt*self.tdrag/self.k


    def setlp(self,lp):
        self.lp = lp


    @property
    def k(self):
        ''' Return the value of the bending modulus, based on the persistance length'''
        return self.lp*4.114

    # Allow user to set the number of points to break the simulation into along the arclength coordinate
    def setNx(self,N):
        ''' Set the number of points to break the simulation into along the arclength, as well as 
        calculate the size of the spacial step in real length
        Args:
                N: The number of points to simulate along the length of the chain
                '''
        self.lcl_dx = self.lc/(N-1)
        self.N = N


    def fastpullProtocol(self,force):
        '''Set the boundry conditions to a jump in force (real force)'''
        self.boundry = np.linspace(force,force,self.numT + 1)
        

    
    def sinProtocol(self,freq,mid,amp):
        '''Set the boundry conditions to a sinusoidal stimulus, with real frequency, force amplitude, and force offset
        Args:
                freq: The real frequency of stimulation
                mid: The offset of the amplitude
                amp: Amplitude of sinusoidal stimulus'''
        t = np.linspace(0,self.numT*self.real_dt,self.numT+1)
        if self.numT*self.real_dt*freq < 10:
           raise RuntimeWarning('Less than ten cycles present, consider changing time step or number of time points to accommodate')
        self.boundry = (mid + amp*np.sin(t*2*np.pi*freq))/self.k
        

    def setValues(self,dt,N,f0,numT):
        '''Set the values for the simulation, including the time step, the number of points along the arclength,
        the initial holding force/prestress, and the number of simulation points
        Args:
                dt: The real timestep
                N: The number of points to simulate along the length of the chain
                f0: The prestress/holding force of the chain before the simulation begins
                numT: The number of timepoints to take in the simulation'''
        self.lcl_dx = self.lc/(N-1)
        self.N = N
        self.real_dt = dt
        self.lcl_dt = dt*self.k/self.tdrag
        self.f0 = f0
        self.numT = int(numT)
  

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
        return matrix/self.lcl_dx**2

  

    

        


    # Feed in the boundry conditions
    
    
    def Simulation(self,numthreads):
        numT = int(self.numT)
        dt = self.lcl_dt
        T = self.real_dt*numT
        J2m = np.zeros((self.N,self.N))
        self.ten_prof = np.zeros((self.N,numT))
        self.sld = np.zeros((self.N,numT))
        f_test = np.zeros(self.N)
        self.F_prof = np.zeros((self.N,numT))
        self.error = np.zeros((self.N,numT))
        F_test = np.zeros(self.N)
        error = np.zeros(self.N)
        P = np.zeros(self.N)
        P1 = np.zeros(self.N)
        P2 = np.zeros(self.N)
        P1J = np.zeros(self.N)
        P2J = np.zeros(self.N)
        tosubtract = np.zeros(self.N)
        self.ten_prof[:,0] = self.f0/self.k
        self.F_prof[:,0] = 0
        clock = np.linspace(0,T,numT+1)
        # INITATE THE TEST VALUE TO START
        f_test = np.linspace(self.boundry[1]/self.k[0],self.boundry[1]/self.k[self.N-1],self.N)

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
                    I = integrate.quad(p1,0,np.inf,args=(j*dt[i],F_test[i],self.f0,self.k[i]))
                    P1[i] = I[0]
            
                    E = integrate.quad(jp1,0,np.inf,args=(j*dt[i],F_test[i],self.f0,self.k[i]))
                    P1J[i] = E[0]


                    #Calculate part 2 of the RHS of equation: need to compute sum of integrals
                    

                    #To calculate which integrals to preform, we will always take the first 50 points and the
                    #0 point. After that, we want to compute the curvature at each point, and if it is above
                    #A certain threshold, we will break the integrals into small pieces around that
                    #point. The curvature cutoff should be based on some sort of exponential, so that
                    #the further in the past we are, the less we care (except for extreme steps), but
                    #for now we will just take an absolute cutoff (of 0.01)
                    parameters = []
                    indexlist = [0]
                    minval = max(0,j-10) 
                    for k in range(minval,j+1): #Always add the last 50 points to the list
                        indexlist.extend([k])
                
                    if j > 10: #AFter the 10th point, begin computing curvature
                        dydx = np.gradient(self.F_prof[i,:j])
                        curvature = np.gradient(dydx) #Compute curvature
                        for k in range(0,len(curvature)): #For each index (j)
                            if abs(curvature[k]) > .01: #If the curvature is above a certain amount
                                que = [max(0,k-2),max(0,k-1),k,min(j,k+1),min(j,k+2)] #Grab all adjacent points
                                for m in que: 
                                    if m not in indexlist: #If the points are not currently in the list
                                        indexlist.extend([m]) #Add them!

                    indexlist = np.unique(indexlist) #np.unique will cut out any duplicates, and arrange the list
                    for k in range(1,len(indexlist)): #Extend the list of paramaters to compute integrals for
                        parameters.extend([[clock[j]*self.k[i]/self.tdrag,clock[int(indexlist[k])]*self.k[i]/self.tdrag,clock[int(indexlist[k-1])]*self.k[i]/self.tdrag,F_test[i],self.F_prof[i,int(indexlist[k])],self.F_prof[i,int(indexlist[k-1])]]])

                    #Compute sum of P2 integrals
                    results_pooled = list(p.map(integral,parameters))
                    results = list(results_pooled)
                    P2[i] = sum(results)

                    #Compute sum of jacobian P2 integrals
                    results_pooled_J = list(p.map(integralJ,parameters))
                    results_J = list(results_pooled_J)
                    P2J[i] = sum(results_J)
            
              
                # Sum the two contributions
                P = P1 + P2
        
                

                for i in range(1,self.N-1):
                    J2m[i,i] = (P1J[i] + P2J[i])*(1/(2*self.lp[i]*np.pi))
            
                Jacobian = self.J1m - J2m
        
   
    
        
                #TAKE 2ND DERIVATIVE OF THE TEST FUNCTION, TO COMPUTE CURVATURE
                dxdy = np.gradient(F_test)
                curvature = (1/self.lcl_dx**2)*np.gradient(dxdy)
                for i in range(0,self.N):
                    if np.abs(curvature[i]) > 1e5:
                        curvature[i] = curvature[i-1]

                curvature = curvature
                error = curvature - (1/(2*self.lp*np.pi))*P
        
                error[0] = 0
                error[self.N - 1] = 0

   
                if ((np.sum(abs(error)))) < .00001:
                    self.ten_prof[:,j] = f_test
                    self.F_prof[:,j] = F_test
                    self.error[:,j] = error
                    self.sld[:,j] = P
                    storage = np.zeros((self.N,j))
                    for k in range(0,j):
                        storage[:,k] = self.ten_prof[:,k]*self.k 
                    print('success')
                    print('Num Trials:',trial)
                    np.savetxt('ten_vals.txt',storage)
                    np.savetxt('f_vals.txt',self.F_prof)
                    np.savetxt('error.txt',self.error)
                    np.savetxt('sld.txt',self.sld)
                    np.savetxt('times.txt',clock)

                    break #Move on to next time step
    
        
        
                tosubtract = (np.dot(np.linalg.pinv(Jacobian),error))
                F_test = F_test - tosubtract
        
             
                f_test = (2/dt)*(F_test - self.F_prof[:,j-1]) - self.ten_prof[:,j-1]
        
        
                f_test[0] = self.boundry[j]/self.k[0]
                f_test[self.N-1] = self.boundry[j]/self.k[self.N-1]

                
        p.terminate()
 
