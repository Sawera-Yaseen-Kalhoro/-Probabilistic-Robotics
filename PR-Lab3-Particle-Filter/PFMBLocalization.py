import numpy as np
import math
from MCLocalization import MCLocalization

class PFMBL(MCLocalization):
    """
    Particle Filter Map Based Localization class.

    This class defines a Map Based Localization using a Particle Filter. It inherits from :class:`MCLocalization`, so the Prediction step is already implemented.
    It needs to implement the Update function, and consecuently the Weight and Resample functions.
    """
    def __init__(self, zf_dim, M, *args) -> None:
        
        self.zf_dim = zf_dim  # dimensionality of a feature observation
        
        self.M = M
        self.nf = len(M)
        super().__init__(*args)


    def Weight(self, z, R): 
        """
        Weight each particle by the liklihood of the particle being correct.
        The probability the particle is correct is given by the probability that it is correct given the measurements (z). 

        
        :param z: measurement vector
        :param R: measurement noise covariance
        :return: None
        """
        # To be completed by the student
        particle_weights = [] # initialize an empty list

        for particle in self.particles:
            # loop for all landmarks in z
            p = 1.0 # initialize probability

            # loop for all landmark in z
            for landmark_index in z.keys():
                landmark = self.M[landmark_index]
                # distance between the particle and the landmark
                d = np.sqrt((particle[0,0]-landmark[0,0])**2 + (particle[1,0]-landmark[1,0])**2)
                
                p *= 1.0/np.sqrt(2*np.pi*R**2) * np.exp(-((d-z[landmark_index])**2)/(2*R**2))
            
            particle_weights.append(p)

        # normalize weights
        particle_weights = particle_weights/sum(particle_weights)

        # assign to attribute
        self.particle_weights = particle_weights

        return
    
    
    def Resample(self):
        """
        Resample the particles based on their weights to ensure diversity and prevent particle degeneracy.

        This function implements the resampling step of a particle filter algorithm. It uses the weights
        assigned to each particle to determine their likelihood of being selected. Particles with higher weights
        are more likely to be selected, while those with lower weights have a lower chance.

        The resampling process helps to maintain a diverse set of particles that better represents the underlying
        probability distribution of the system state. 

        After resampling, the attributes 'particles' and 'weights' of the ParticleFilter instance are updated
        to reflect the new set of particles and their corresponding weights.

        :return: None
        """
        # To be completed by the student

        # using the algorithm for low variance sampler
        new_particles = [] #initialize list of new particles
        new_particle_weights = [] #initialize list of new particle weights

        W = sum(self.particle_weights)
        M = len(self.particles)
        r = np.random.uniform(0,W/M)
        c = self.particle_weights[0]
        i = 0

        for m in range(1,M+1):
            u = r + (m-1)*W/M
            while u > c:
                i += 1
                c += self.particle_weights[i]
            new_particles.append(self.particles[i])
            new_particle_weights.append(1/M)
        
        # assign to attributes
        self.particles = new_particles
        self.particle_weights = new_particle_weights

        return
    
    def Update(self, z, R):
        """
        Update the particle weights based on sensor measurements and perform resampling.

        This function adjusts the weights of particles based on how well they match the sensor measurements.
       
        The updated weights reflect the likelihood of each particle being the true state of the system given
        the sensor measurements.

        After updating the weights, the function may perform resampling to ensure that particles with higher
        weights are more likely to be selected, maintaining diversity and preventing particle degeneracy.
        
        :param z: measurement vector
        :param R: the covariance matrix associated with the measurement vector

        """
        # To be completed by the student
        
        self.Weight(z,R)

        # define condition to resample
        self.n_eff = 1/sum([w**2 for w in self.particle_weights])
        if self.n_eff < len(self.particles)/2:
            self.Resample()
        
        return
    

    def Localize(self):
        uk, Qk = self.GetInput()

        if uk.size > 0:
            self.Prediction(uk, Qk)
        
        zf, Rf = self.GetMeasurements()
        # if zf.size > 0:
        if len(zf) > 0:
            self.Update(zf, Rf)

        self.PlotParticles()
        return self.get_mean_particle()