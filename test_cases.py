"""
Test cases for the quick implementation of the Constitutive model for UHMWPE by
J.S. Bergstrom and J.E. Bischoff, IJSCS Vol 2:1, 2010, pp. 31-39

Developed using:
- Python 2.7
- Numpy 1.9.3

jibril.coulibaly at gmail.com

all units SI

"""

import numpy as np
import quick
import matplotlib.pyplot as plt

################################################################################
# Uniaxial tension / compression at different loading rates (after Figure 4a)
################################################################################

nsave = 500 # Number of data points saved on each loading path, [-]
dt = 0.01 # Timestep of the explicit integration (must be small), [s]

# Uniaxial tension
# - Engineering strain rate = 0.007 [1/s], true strain = 1.25 [-]
f_t7, sig_t7 = quick.main(0.007, np.array([0.0, 1.25]), dt, nsave)
# - Engineering strain rate = 0.018 [1/s], true strain = 1.25 [-]
f_t18, sig_t18 = quick.main(0.018, np.array([0.0, 1.25]), dt, nsave)
# - Engineering strain rate = 0.035 [1/s], true strain = 1.25 [-]
f_t35, sig_t35 = quick.main(0.035, np.array([0.0, 1.25]), dt, nsave)

# Uniaxial compression
# - Engineering strain rate = 0.010 [1/s], true strain = -1 [-]
f_c, sig_c = quick.main(0.010, np.array([0.0, -1.0]), dt, nsave)


# Plot results
plt.plot(np.log(f_t7[:,0]),sig_t7[:,0]*1e-6,label="Tension, erate=0.007/s")
plt.plot(np.log(f_t18[:,0]),sig_t18[:,0]*1e-6,label="Tension, erate=0.018/s")
plt.plot(np.log(f_t35[:,0]),sig_t35[:,0]*1e-6,label="Tension, erate=0.035/s")
plt.plot(np.log(f_c[:,0]),sig_c[:,0]*1e-6,label="Compression, erate=0.010/s")
plt.xlabel("True strain, [-]")
plt.ylabel("True stress, [MPa]")
plt.legend()

################################################################################
# Cyclic loading at different loading rates (after Figure 4b)
################################################################################

# Extension - contraction
# - Engineering strain rate = 0.002 [1/s], true strain = [0,0.15, 0.06] [-]
f_tc2, sig_tc2 = quick.main(0.002, np.array([0.0, 0.15, 0.06]), dt, nsave)

# Cyclic loading
# - Engineering strain rate = 0.005 [1/s]
# - True strain = [0,0.11, -0.12, 0.11, -0.12] [-]
f_cyc5, sig_cyc5 = quick.main(0.005, np.array([0, 0.11, -0.12, 0.11, -0.12]),
                              dt, nsave)

# Contraction - extension
# - Engineering strain rate = 0.002 [1/s], true strain = [0,-0.42,-0.31] [-]
f_ct2, sig_ct2 = quick.main(0.002, np.array([0,-0.42,-0.31]), dt, nsave)
# - Engineering strain rate = 0.005 [1/s], true strain = [0,-0.42,-0.31] [-]
f_ct5, sig_ct5 = quick.main(0.005, np.array([0,-0.42,-0.31]), dt, nsave)
# - Engineering strain rate = 0.010 [1/s], true strain = [0,-0.42,-0.31] [-]
f_ct10, sig_ct10 = quick.main(0.01, np.array([0,-0.42,-0.31]), dt, nsave)

plt.plot(np.log(f_tc2[:,0]),sig_tc2[:,0]*1e-6,label="erate=0.002/s")
plt.plot(np.log(f_cyc5[:,0]),sig_cyc5[:,0]*1e-6,label="erate=0.005/s")
plt.plot(np.log(f_ct2[:,0]),sig_ct2[:,0]*1e-6,label="erate=0.002/s")
plt.plot(np.log(f_ct5[:,0]),sig_ct5[:,0]*1e-6,label="erate=0.005/s")
plt.plot(np.log(f_ct10[:,0]),sig_ct10[:,0]*1e-6,label="erate=0.010/s")
plt.xlabel("True strain, [-]")
plt.ylabel("True stress, [MPa]")
plt.legend()
