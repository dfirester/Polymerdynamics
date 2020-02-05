import tensionDynamics
from tensionDynamics import Polymer
import numpy as np 

chain = Polymer(1,1)
x = np.linspace(0,10000,10001)
boundry = 5*np.sin(x/100) + 5
chain.setValues(.1,100,1,boundry)

chain.Simulation(1000,60)

np.savetxt('ten_vals.txt',chain.ten_prof)
np.savetxt('f_vals.txt',chain.F_prof)