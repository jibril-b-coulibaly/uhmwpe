"""
Quick implementation of the thermomechanical Constitutive model for UHMWPE by
J.S. Bergstrom and J.E. Bischoff, IJSCS Vol 2:1, 2010, pp. 31-39

- Parts related to temperature are not implemented as omitted in the paper

Single material point
Explicit integration (Linearized steps Bardet and Choucair)
Finite differences Jacobian

Developed using:
- Python 2.7
- Numpy 1.9.3

jibril.coulibaly at gmail.com

all units SI
"""

import numpy as np
import matplotlib.pyplot as plt

################################################################################
# Material parameters (Table 1 p. 36)
################################################################################
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

################################################################################
# Utility functions and constants
################################################################################

ONETHIRD = 1./3
TWOTHIRD = 2./3

id33 = np.diag([1.0, 1.0, 1.0]) # 3x3 Identity matrix
idV = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]) # Identity matrix, Voigt notation

# Compute elastic deformation gradient from total and viscoplastic part
# Solve F = Fe * Fv for Fe. Transpose to express as Ax=B: Fv^T * Fe^T = F^T
def computeFe33(f, fv):
    return np.transpose(np.linalg.solve(np.transpose(fv), np.transpose(f)))

# Compute elastic left Cauchy-Green tensor from elastic deformation gradient
# Stored as 6x1 Voigt vector. Also return Jacobian of elastic transformation Je
def computeBeV(fe):
    je = np.linalg.det(fe)
    be33 = je**(-TWOTHIRD)*np.dot(fe, np.transpose(fe))
    return [je, np.array([be33[0,0], be33[1,1], be33[2,2],
                          be33[1,2], be33[0,2], be33[0,1]])]

# Compute the effective chain stretch from elastic left Cauchy-Green tensor
def computeLam(be):
    return np.sqrt(ONETHIRD*(be[0] + be[1] + be[2]))

# Approximate of inverse Langevin function according to Jedynak 2015
# doi:10.1007/s00397-014-0802-2
# see also https://en.wikipedia.org/wiki/Brillouin_and_Langevin_functions
# ONLY VALID FOR POSITIVE VALUES x>0, which is the case for this model
def invLang(x):
    return x*(3.0 - 2.6*x + 0.7*x*x)/((1 - x)*(1 + 0.1*x))

# Compute deviatoric part of symmetric tensor in Voigt notation
def devV(tV):
    return tV - ONETHIRD*(tV[0] + tV[1] + tV[2])*idV

# Compute the invariants I1*, I2* of the elastic left Cauchy-Green tensor
def computeInvar(be):
    i1 = np.trace(be)
    i2 = 0.5*(np.trace(be)**2 - np.trace(np.dot(be,be)))
    return [i1, i2]


# Variables and initial values


f = np.diag # Total deformation gradient
fvA = id3 # Viscoplastic deformation gradient of A
fvB = id3 # Viscoplastic deformation gradient of B