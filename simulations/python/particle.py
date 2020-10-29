class Particle():
    def __init__(self, source, E_k, M, N=1):
        self.source = source
        particles = source.GenerateParticle(E_k, M, N)
        self.particle_x = particles[1][:,0]
        self.particle_y = particles[1][:,1]
        self.particle_z = particles[1][:,2]
        self.particle_px = particles[4][:,0]
        self.particle_py = particles[4][:,1]
        self.particle_pz = particles[4][:,2]

    def FreeEvolution(self):
        pass

    def TimeEvolution(self, F, dt):
        # x_(i+1) = x_i + t_i*px_i/M
        # px_(i+1) = px_i + F_i*t_i
        pass
