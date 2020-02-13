import tensionDynamics
from tensionDynamics import Polymer
import numpy as np 

chain = Polymer(150,1e-7)

lpvals = np.linspace(10,10,50)
lpvals = np.append(lpvals, np.linspace(20,20,50))
chain.setlp(lpvals)

chain.setValues(1e-10,100,1,10)

chain.fastpullProtocol(10)

chain.Simulation(1)

