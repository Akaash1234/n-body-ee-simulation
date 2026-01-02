# lagrange.py
# CR3BP lagrange point stuff
# had to read sm papers for this lol

import numpy as np
from scipy.optimize import brentq

def lagrange_pts(mu):
    """
    find L1-L5
    mu = m2/(m1+m2)
    """
    pts = {}
    
    # L4 L5 are ez - equilateral triangles
    pts['L4'] = (0.5 - mu, np.sqrt(3)/2)
    pts['L5'] = (0.5 - mu, -np.sqrt(3)/2)
    
    # L1 L2 L3 need solving, pain
    # got these equations from somewhere idr
    
    def f_L1(x):
        return x - (1-mu)/(x+mu)**2 + mu/(x-1+mu)**2
    
    def f_L2(x):
        return x - (1-mu)/(x+mu)**2 - mu/(x-1+mu)**2
    
    def f_L3(x):
        return x + (1-mu)/(x+mu)**2 + mu/(x-1+mu)**2
    
    try:
        pts['L1'] = (brentq(f_L1, -mu+0.01, 1-mu-0.01), 0)
    except:
        pts['L1'] = (0.8369 - mu, 0)  # approx wtv
    
    try:
        pts['L2'] = (brentq(f_L2, 1-mu+0.01, 2), 0)
    except:
        pts['L2'] = (1.1557 - mu, 0)
    
    try:
        pts['L3'] = (brentq(f_L3, -2, -mu-0.01), 0)
    except:
        pts['L3'] = (-1.0051, 0)
    
    return pts

def effective_potential(x, y, mu):
    """pseudo-potential, rotating frame"""
    r1 = np.sqrt((x + mu)**2 + y**2)
    r2 = np.sqrt((x - 1 + mu)**2 + y**2)
    # sign convention is confusing, lw sure this is right
    return -0.5*(x**2 + y**2) - (1-mu)/r1 - mu/r2

def potential_grid(mu, xlim=(-1.5, 1.5), ylim=(-1.5, 1.5), res=100):
    x = np.linspace(*xlim, res)
    y = np.linspace(*ylim, res)
    X, Y = np.meshgrid(x, y)
    Z = effective_potential(X, Y, mu)
    Z = np.clip(Z, -10, 0)  # clip near masses
    return X, Y, Z

def critical_mu():
    # routh criterion, L4/L5 stable below this
    return (1 - np.sqrt(23/27)) / 2  # ~0.0385

def is_stable(mu):
    return mu < critical_mu()

def jacobi_constant(x, y, vx, vy, mu):
    """conserved in CR3BP"""
    U = effective_potential(x, y, mu)
    v2 = vx**2 + vy**2
    return -2 * U - v2

def stability_eigenvalues(mu, L_point='L4'):
    # characteristic eqn: λ^4 + λ^2 + (27/4)μ(1-μ) = 0
    # idrc about L1-L3 rn just L4
    
    a = 1
    b = 1
    c = (27/4) * mu * (1 - mu)
    
    disc = b**2 - 4*a*c
    if disc >= 0:
        s1 = (-b + np.sqrt(disc)) / (2*a)
        s2 = (-b - np.sqrt(disc)) / (2*a)
    else:
        s1 = complex(-b/(2*a), np.sqrt(-disc)/(2*a))
        s2 = complex(-b/(2*a), -np.sqrt(-disc)/(2*a))
    
    eigs = []
    for s in [s1, s2]:
        if isinstance(s, complex) or s < 0:
            lam = np.sqrt(complex(s))
            eigs.extend([lam, -lam])
        else:
            lam = np.sqrt(s)
            eigs.extend([lam, -lam])
    return eigs

def hill_sphere(a, m_planet, m_star):
    return a * (m_planet / (3 * m_star)) ** (1/3)

def zero_velocity_curves(mu, C_J, xlim=(-2,2), ylim=(-2,2), n=200):
    x = np.linspace(*xlim, n)
    y = np.linspace(*ylim, n)
    X, Y = np.meshgrid(x, y)
    
    U = effective_potential(X, Y, mu)
    C_surface = -2 * U
    
    allowed = C_surface >= C_J
    return X, Y, C_surface, allowed


# === CR3BP dynamics ===

def cr3bp_acceleration(x, y, vx, vy, mu):
    """accel in rotating frame w/ coriolis"""
    r1 = np.sqrt((x + mu)**2 + y**2) + 1e-10
    r2 = np.sqrt((x - 1 + mu)**2 + y**2) + 1e-10
    
    ax_grav = -(1-mu)*(x + mu)/r1**3 - mu*(x - 1 + mu)/r2**3
    ay_grav = -(1-mu)*y/r1**3 - mu*y/r2**3
    
    # centrifugal + coriolis
    ax = x + ax_grav + 2*vy
    ay = y + ay_grav - 2*vx
    
    return ax, ay


def simulate_cr3bp(x0, y0, vx0, vy0, mu, dt, steps):
    """test particle sim in rotating frame"""
    x, y, vx, vy = x0, y0, vx0, vy0
    xs = [x]
    ys = [y]
    jacobi = [jacobi_constant(x, y, vx, vy, mu)]
    
    # rk4, copied from solver lol
    for step in range(steps):
        ax1, ay1 = cr3bp_acceleration(x, y, vx, vy, mu)
        
        x2 = x + 0.5*dt*vx
        y2 = y + 0.5*dt*vy
        vx2 = vx + 0.5*dt*ax1
        vy2 = vy + 0.5*dt*ay1
        ax2, ay2 = cr3bp_acceleration(x2, y2, vx2, vy2, mu)
        
        x3 = x + 0.5*dt*vx2
        y3 = y + 0.5*dt*vy2
        vx3 = vx + 0.5*dt*ax2
        vy3 = vy + 0.5*dt*ay2
        ax3, ay3 = cr3bp_acceleration(x3, y3, vx3, vy3, mu)
        
        x4 = x + dt*vx3
        y4 = y + dt*vy3
        vx4 = vx + dt*ax3
        vy4 = vy + dt*ay3
        ax4, ay4 = cr3bp_acceleration(x4, y4, vx4, vy4, mu)
        
        x += dt/6 * (vx + 2*vx2 + 2*vx3 + vx4)
        y += dt/6 * (vy + 2*vy2 + 2*vy3 + vy4)
        vx += dt/6 * (ax1 + 2*ax2 + 2*ax3 + ax4)
        vy += dt/6 * (ay1 + 2*ay2 + 2*ay3 + ay4)
        
        xs.append(x)
        ys.append(y)
        jacobi.append(jacobi_constant(x, y, vx, vy, mu))
        
        # if abs(jacobi[-1] - jacobi[0]) > 0.01:
        #     print(f"jacobi drift at {step}")  # debug

    return np.array(xs), np.array(ys), np.array(jacobi)

def flow_field(mu, xlim=(-1.5, 1.5), ylim=(-1.5, 1.5), n=20):
    """accel vector field at v=0"""
    x = np.linspace(*xlim, n)
    y = np.linspace(*ylim, n)
    X, Y = np.meshgrid(x, y)
    
    AX, AY = cr3bp_acceleration(X, Y, np.zeros_like(X), np.zeros_like(Y), mu)
    
    M = np.hypot(AX, AY) + 1e-5
    AX /= M
    AY /= M
    
    return X, Y, AX, AY

def poincare_section(mu, C_J, max_iter=1000):
    """
    poincare at y=0
    kinda slow but wtv
    """
    points = []
    
    for x0 in np.linspace(-0.8, 0.8, 30):
        y0 = 0
        U = effective_potential(x0, y0, mu)
        v2 = -2*U - C_J
        
        if v2 > 0:
            vy0 = np.sqrt(v2)
            vx0 = 0
            
            x, y, vx, vy = x0, y0, vx0, vy0
            dt = 0.01
            
            # euler here, idrc about accuracy for poincare
            for _ in range(max_iter):
                ax, ay = cr3bp_acceleration(x, y, vx, vy, mu)
                x += vx * dt
                y += vy * dt
                vx += ax * dt
                vy += ay * dt
                
                if abs(y) < 0.05 and vy > 0:
                    points.append([x, vx])
                    
    return np.array(points) if points else np.array([])

