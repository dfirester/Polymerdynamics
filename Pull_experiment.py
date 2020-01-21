import tensionDynamics
from tensionDynamics import Polymer
import numpy as np 

chain = Polymer(1,1)
boundry = np.linspace(10,10,2000)
chain.setValues(.1,100,1,boundry)

chain.Simulation(10,50)

np.savetxt('ten_vals.txt',chain.ten_prof)
np.savetxt('f_vals.txt',chain.F_prof)