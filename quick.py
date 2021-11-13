"""
Quick implementation of the thermomechanical Constitutive model for UHMWPE by
J.S. Bergstrom and J.E. Bischoff, IJSCS Vol 2:1, 2010, pp. 31-39

- Parts related to temperature are not implemented as omitted in the paper
- Symmetric tensors stored as 3x3 and not Voigt for simple use of Python linalg

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

# Compute elastic deformation gradient from total and viscoplastic part
# Solve F = Fe * Fv for Fe. Transpose to express as Ax=B: Fv^T * Fe^T = F^T
def computeFe(f, fv):
    return np.transpose(np.linalg.solve(np.transpose(fv), np.transpose(f)))

# Compute elastic left Cauchy-Green tensor from deformation gradient
# Compute the invariants I1*, I2*, J of the left Cauchy-Green tensor
def computeBeInvar(f):
    j = np.linalg.det(f)
    b = j**(-TWOTHIRD)*np.dot(f, np.transpose(f))
    i1 = np.trace(b)
    i2 = 0.5*(np.trace(b)**2 - np.trace(np.dot(b,b)))
    return b, i1, i2, j

# Compute the effective chain stretch from left Cauchy-Green tensor
def computeLam(b):
    return np.sqrt(ONETHIRD*np.trace(b))

# Approximate of inverse Langevin function according to Jedynak 2015
# doi:10.1007/s00397-014-0802-2
# see also https://en.wikipedia.org/wiki/Brillouin_and_Langevin_functions
# ONLY VALID FOR POSITIVE VALUES x>0, which is the case for this model
def invLang(x):
    return x*(3.0 - 2.6*x + 0.7*x*x)/((1 - x)*(1 + 0.1*x))

# Compute deviatoric part of tensor
def dev(t):
    return t - ONETHIRD*np.trace(t)*id33

# Compute stress for networks A and B (equation (1) p.32, and (2) p.33)
# the terms related to temperature are ignored
def computeStressAB(fe, mu, k, lamL):
    be, je = computeBeInvar(fe)[0::3]
    lam = computeLam(be)
    lamLinv = 1./lamL
    return (mu/(je*lam)*invLang(lam*lamLinv)/invLang(lamLinv)*dev(be) +
            k*(je-1)*id33)

# Compute stress for networks C (equation (4) p.34)
# the terms related to temperature are ignored
def computeStressC(f, mu, k, q, lamL):
    b, i1, i2, j = computeBeInvar(f)
    return ((1./(1 + q))*(computeStressAB(f, mu, k, lamL) +
                          q*mu/j*(i1*b - TWOTHIRD*i2*id33 - np.dot(b,b))))

# Compute the velocity gradient of the viscoleastic flow for networks A and B
# Equations (5), (6) p.35 and equations (7), (8) p.35-36
# Solve Fvdot = Fe^-1 *gamma*(dev(sigma)/tau)*F. for Fvdot. Left multiply by Fe
# to solve linear system of the form AX=B instead of computing the inverse:
# Fe*Fvdot = gamma*(dev(sigma)/tau)*F, to solve for Fv
# the terms related to temperature are ignored
def computeFvdot(f, fe, sig, that, a, m):
    s = dev(sig)
    p = -ONETHIRD*np.trace(sig)
    t = np.linalg.norm(s)
    g = (t/(that + a*0.5*(p+np.abs(p))))**m
    return np.linalg.solve(fe, g/t*np.dot(s, f))




# Variables and initial values


f = np.diag # Total deformation gradient
fvA = id3 # Viscoplastic deformation gradient of A
fvB = id3 # Viscoplastic deformation gradient of B