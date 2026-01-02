import numpy as np

G = 1.0

class Sim:
    def __init__(self, pos, vel, mass):
        self.pos = np.array(pos, dtype=float)
        self.vel = np.array(vel, dtype=float)
        self.m = np.array(mass, dtype=float)
        self.n = len(mass)
        self.g = G

    def forces(self):
        acc = np.zeros_like(self.pos)
        for i in range(self.n):
            for j in range(self.n):
                if i == j: continue
                r = self.pos[j] - self.pos[i]
                d = np.linalg.norm(r) + 1e-10
                acc[i] += self.g * self.m[j] * r / d**3
        return acc

    def step_euler(self, dt):
        acc = self.forces()
        self.vel += acc * dt
        self.pos += self.vel * dt

    def step_verlet(self, dt):
        acc = self.forces()
        self.vel += 0.5 * acc * dt
        self.pos += self.vel * dt
        acc_new = self.forces()
        self.vel += 0.5 * acc_new * dt

    def step_leapfrog(self, dt):
        self.vel += 0.5 * dt * self.forces()
        self.pos += dt * self.vel
        self.vel += 0.5 * dt * self.forces()

    def step_rk4(self, dt):
        def deriv(p, v):
            tmp = Sim(p, v, self.m)
            tmp.g = self.g
            return v.copy(), tmp.forces()

        k1v, k1a = deriv(self.pos, self.vel)
        k2v, k2a = deriv(self.pos + 0.5*dt*k1v, self.vel + 0.5*dt*k1a)
        k3v, k3a = deriv(self.pos + 0.5*dt*k2v, self.vel + 0.5*dt*k2a)
        k4v, k4a = deriv(self.pos + dt*k3v, self.vel + dt*k3a)

        self.pos += dt/6 * (k1v + 2*k2v + 2*k3v + k4v)
        self.vel += dt/6 * (k1a + 2*k2a + 2*k3a + k4a)

    def step(self, dt, method='verlet'):
        if method == 'euler': self.step_euler(dt)
        elif method == 'rk4': self.step_rk4(dt)
        elif method == 'leapfrog': self.step_leapfrog(dt)
        else: self.step_verlet(dt)

    def run(self, dt, steps, method='verlet'):
        hist = [{'pos': self.pos.copy(), 'vel': self.vel.copy()}]
        for i in range(steps):
            self.step(dt, method)
            hist.append({'pos': self.pos.copy(), 'vel': self.vel.copy()})
        return hist
