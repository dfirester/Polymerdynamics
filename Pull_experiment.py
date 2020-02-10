import tensionDynamics
from tensionDynamics import Polymer
import numpy as np 

chain = Polymer(10,150,1e-7)
chain.setValues(10,101,1,50000)
chain.sinProtocol(1e5,5,5)
chain.Simulation(60)

