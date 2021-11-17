"""
Quick implementation of the thermomechanical Constitutive model for UHMWPE by
J.S. Bergstrom and J.E. Bischoff, IJSCS Vol 2:1, 2010, pp. 31-39

- Parts related to temperature are not implemented as omitted in the paper
- Symmetric tensors stored as 3x3 and not Voigt for simple use of Python linalg

- Single material point under uniaxial tension/compression
- Explicit forward difference integration
- Finite differences Jacobian

Developed using:
- Python 2.7
- Numpy 1.9.3

jibril.coulibaly at gmail.com

all units SI

TODO:
- Space optimization using Voigt representation for symmetric tensors
"""

import numpy as np
import matplotlib.pyplot as plt

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

# Compute total stress
def computeStress(f, feA, feB, muA, muB, muC, k, q, lamL):
    return (computeStressAB(feA, muA, k, lamL) +
            computeStressAB(feB, muB, k, lamL) +
            computeStressC(f, muC, k, q, lamL))

# Compute the velocity gradient of the viscoleastic flow for networks A and B
# Equations (5), (6) p.35 and equations (7), (8) p.35-36
# Solve Fvdot = Fe^-1 *gamma*(dev(sigma)/tau)*F. for Fvdot. Left multiply by Fe
# to solve linear system of the form AX=B instead of computing the inverse:
# Fe*Fvdot = gamma*(dev(sigma)/tau)*F, to solve for Fv
# the terms related to temperature are ignored
# Also return gamma for shear modulus evolution equation (3) p.33
def computeFvdot(f, fe, sig, that, a, m):
    s = dev(sig)
    p = -ONETHIRD*np.trace(sig)
    t = np.linalg.norm(s)
    g = (t/(that + a*0.5*(p+np.abs(p))))**m
    #if (g >= 1.0):
    #    g = 1.0
    #else:
    #    g = g**m
    if (t <= 0):
        fvdot = np.zeros((3,3))
    else:
        fvdot = np.linalg.solve(fe, g/t*np.dot(s, f))
    return fvdot, g

# Numerical computation of the tangent sitffness matrix
# Derivative of stress with respect to deformation gradient
# Time-dependent material -> time-dependent stiffness matrix computed for:
# - an evolution of the viscoplastic part over 1 timestep
# - a deformation gradient step consistent with the applied strain rate
# Store in a 9x9 matrix for simplicity (row-major storage of 3x3 tensors)
# We could take advantage of the symetry of cauchy stress and store 6x9
def tangent(sig_t, f_t, fvA_tplusdt, fvB_tplusdt,
            erate, dt, muA, muB_tplusdt, k, q, lamL):
    dsig_dF = np.zeros((9,9))
    for i in range(3):
        for j in range(3):
            # F = 1 + H, displacement gradient increment dH=erate*dt
            f_tplusdt = np.copy(f_t)
            f_tplusdt[i][j] += erate*dt
            # Computation of the elastic deformation gradients at t+dt
            feA_tplusdt = computeFe(f_tplusdt, fvA_tplusdt)
            feB_tplusdt = computeFe(f_tplusdt, fvB_tplusdt)
            # Computation of the stress at t+dt
            sig_tplusdt = computeStress(f_tplusdt, feA_tplusdt, feB_tplusdt,
                                        muA, muB_tplusdt, muC, k, q, lamL)
            # Compute element-wise stiffness Finite difference
            dsig_dF[:,i*3+j] = (sig_tplusdt - sig_t).flatten()/(erate*dt)
    return dsig_dF

################################################################################
# Computation of the isothermal mechanical behavior for a single material point
################################################################################

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

# Loading and data saving conditions, from command line
erate = 0.007 # Engineering strain rate, [1/s]
etrue = np.array([0.0, 1.25]) # List of values of true strain imposed loading, [-]
eeng = np.exp(etrue) - 1.0 # Convert to engineering strain, [-]
deeng = eeng[1:] - eeng[0:-1] # Variation of engineering strain on each path
npath = len(deeng) # Number of distinct loading paths
dt = 0.1 # Timestep (must meet CFL condition), [s]
nsave = 100 # Number of steps to save on each loading path


# int(sys.argv[11])

# Variables, initial values and saved values
sig = np.zeros((3,3)) # Total Cauchy stress
sigA = np.zeros((3,3)) # Cauchy stress of network A
sigB = np.zeros((3,3)) # Cauchy stress of network B
muB = muBi # Shear modulus of network B
f = np.copy(id33) # Total deformation gradient
feA = np.copy(id33) # Elastic deformation gradient of network A
fvA = np.copy(id33) # Viscoplastic deformation gradient of network A
feB = np.copy(id33) # Elastic deformation gradient of network B
fvB = np.copy(id33) # Viscoplastic deformation gradient of network B
sig_save = np.zeros((npath*nsave,9)) # Cauchy stress, [Pa]
sigA_save = np.zeros((npath*nsave,9)) # Cauchy stress, [Pa]
f_save = np.zeros((npath*nsave,9)) # Deformation gradient, [-]


for ipath in range(npath):
    # Number of steps on loading path (rounded above)
    nstep = np.ceil(deeng[ipath]/(erate*dt)).astype('int')
    nevery = np.round(np.linspace(0,nstep,nsave))
    for istep in range(nstep):
        t = ipath*nstep + istep # Current timestep
        flag = nevery == t
        if (np.any(flag)):
            idx = np.where(flag)[0][0]
            sig_save[idx] = sig.flatten()
            sigA_save[idx] = sigA.flatten()
            f_save[idx] = f.flatten()

        # Explicit time-integration with Forward Euler difference
        # All quantities known at time t

        # Computation of viscoplastic deformation gradients at time t+dt
        fvAdot, ga = computeFvdot(f, feA, sigA, thatA, a, mA)
        fvA = fvA + fvAdot*dt
        fvB = fvB + computeFvdot(f, feB, sigB, thatB, a, mB)[0]*dt
        # Computation of the shear modulus of network B of at time t+dt
        muB = muB -beta*(muB-muBf)*ga*dt

        # Computation of the tangent operator at time t
        ds_df = tangent(sig, f, fvA, fvB, erate, dt, muA, muB, kap, q, lamL)

        # Determination of the total deformation gradient at time t+dt
        # The 9x9 tangent operator is singular since Cauchy stress is symmetric
        # -> cannot solve with classical Newton-Raphson.
        # Symmetry decreases to 6x9 which cannot be solved in general, i.e.,
        # 6 equations for 9 variables -> solution not unique
        ds_df69 = ds_df[[0,1,2,4,5,8],:] # Delete duplicate extradiagonal
        # We must solve for a particular direction (directional derivative).
        # We use a predictor - corrector method
        # Predictor following the imposed deformation gradient F11 only
        df_pre = np.array([erate*dt,0,0,0,0,0,0,0,0])
        dsig_pre = np.dot(ds_df69, df_pre)
        # Corrector in diagonal directions F22 and F33
        dsig_cor = -sig.flatten()[[4,8]] - dsig_pre[[3,5]]
        df_cor = np.linalg.solve(ds_df69[np.ix_([3,5],[4,8])], dsig_cor)
        df = np.copy(df_pre) + np.array([0,0,0,0,df_cor[0],0,0,0,df_cor[1]])

        f = f + df.reshape(3,3)
        # Elastic deformation gradients at t+dt
        feA = computeFe(f, fvA)
        feB = computeFe(f, fvB)
        # Computation of the stresses at time t+dt
        sigA = computeStressAB(feA, muA, kap, lamL)
        sigB = computeStressAB(feB, muB, kap, lamL)
        sig = computeStress(f, feA, feB, muA, muB, muC, kap, q, lamL)

# Save final state
sig_save[-1] = sig.flatten()
f_save[-1] = f.flatten()


plt.plot(np.log(f_save[:,0]),sig_save[:,0]*1e-6,label="erate="+str(erate)+"/s")
plt.xlabel("True strain, [-]")
plt.ylabel("True stress, [MPa]")
plt.legend()
