import math

class Model():
    def __init__(self):
        self.m = 0.5   # kg
        self.l0 = 1     # m
        self.k = 2000   # spring constant
        self.g = 9.81  # m/s^2
        self.theta = -math.pi/50

    def flight_dynamics(self, t, s : list) -> list:
        x, y, xdot, ydot = s
        xddot = 0.0
        yddot = -self.g
        return [xdot, ydot, xddot, yddot] 
    
    def stance_dynamics(self, t, s:list, x_c) -> list:
        x, y, xdot, ydot = s

        dx = x - x_c   # horizontal from foot to COM

        l = math.hypot(dx, y)
        
        # Spring force
        F = self.k * (self.l0 - l)

        # accelerations
        xddot = (F / (self.m * l)) * dx
        yddot = (F / (self.m * l)) * y - self.g


        return [xdot, ydot, xddot, yddot]   # ds/dt

