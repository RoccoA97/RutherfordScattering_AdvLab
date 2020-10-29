import numpy as np
from constants import *




class Source():
    def __init__(self, s_x, s_y, s_z, s_r, s_a):
        self.source_x = s_x    # x position
        self.source_y = s_y    # y position
        self.source_z = s_z    # z position
        self.source_r = s_r    # radius
        self.source_a = s_a    # activity

    def GenerateParticle(self, E_k, M, N=1):
        # kinematics
        gamma = 1.0 + E_k/M
        beta  = np.sqrt(1 - 1/(gamma*gamma))
        p     = M * gamma * beta

        # source point of emission (polar coordinates)
        r_s = np.sqrt(np.random.uniform(0, 1, size=(N,1))) * self.source_r
        t_s = np.random.uniform(0, 1, size=(N,1)) * 2 * np.pi

        # source point of emission (carthesian coordinates)
        x_s = self.source_x + r_s * np.cos(t_s)
        y_s = self.source_y + r_s * np.sin(t_s)
        z_s = self.source_z * np.ones((N,1))

        # versor of direction (polar coordinates)
        u_pp = np.random.uniform(0, 1, size=(N,1)) * 2 * np.pi             # momentum: phi
        u_pt = np.arccos(2 * np.random.uniform(0, 1, size=(N,1)) - 1.0)    # momentum: theta

        # versor of direction (carthesian coordinates)
        u_px = np.cos(u_pp)*np.sin(u_pt)    # momentum : x
        u_py = np.sin(u_pp)*np.sin(u_pt)    # momentum : y
        u_pz = np.cos(u_pt)                 # momentum : z

        # momentum of alphas (carthesian coordinates)
        px = u_px * p
        py = u_py * p
        pz = u_pz * p

        return [np.hstack((r_s,t_s)),
                np.hstack((x_s,y_s,z_s)),
                np.hstack((r_s,t_s)),
                np.hstack((u_px,u_py,u_pz)),
                np.hstack((px,py,pz))]
