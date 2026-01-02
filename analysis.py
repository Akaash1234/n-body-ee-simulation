import numpy as np

def kinetic(vel, m):
    return 0.5 * np.sum(m[:, None] * vel**2)

def potential(pos, m, G=1.0):
    n = len(m)
    u = 0
    for i in range(n):
        for j in range(i+1, n):
            r = np.linalg.norm(pos[j] - pos[i]) + 1e-10
            u -= G * m[i] * m[j] / r
    return u

def total_energy(pos, vel, m, G=1.0):
    return kinetic(vel, m) + potential(pos, m, G)

def angular_momentum(pos, vel, m):
    L = np.zeros(3)
    for i in range(len(m)):
        L += m[i] * np.cross(pos[i], vel[i])
    return L

def energy_error(e0, e):
    if e0 == 0: return 0
    return abs(e - e0) / abs(e0)
