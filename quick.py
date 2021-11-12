"""
Quick implementation of the thermomechanical Constitutive model for UHMWPE by
J.S. Bergstrom and J.E. Bischoff, IJSCS Vol 2:1, 2010, pp. 31-39

Single material point
Explicit integration
Finite differences 

jibril.coulibaly at gmail.com

all units SI
"""

import numpy as np

# Material parameters (Table 1 p. 36)
muA = 200e6 # Shear modulus of network A, [Pa]
lamL = 3.25 # Locking stretch, [-]
kap = 6000e6 # Bulk modulus, [Pa]
thatA = 3.25e6 # Flow resistance of network A, [Pa]
a = 0.073 # Pressure dependence of flow, [-]
mA = mB = 20 # Stress exponential of network A/B, [-]
muBi = 293e6 # Initial shear modulus of network B, [Pa]
muBf = 79.1e6 # Final shear modulus of network B, [Pa]
beta = 31.9 # Evolution rate of muB, [-]
thatB = 20.1e6 # Flow resistance of network B, [Pa]
muC = 10.0e6 # Shear modulus of network C, [Pa]
q = 0.23 #Relatiive contribution of I2 of network C




# Variables