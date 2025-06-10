import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lti, lsim, welch
import torch
import torch.fft as fft

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


import numpy as np

class StephanModelEq6:
    def __init__(self):
        # Define model parameters
        self.k = 0.64
        self.chi = 0.4
        self.TE = 0.04
        self.tau = 1.0
        self.alpha = 0.32
        self.E0 = 0.4
        self.V0 = 0.04
        self.r0 = 25
        self.theta0 = 40.3
        self.epsilon = 1

        # Derived parameters
        self.k1 = 4.3 * self.theta0 * self.E0 * self.TE
        self.k2 = 0.3 * self.r0 * self.E0 * 0.04
        self.k3 = 1 - self.epsilon
        self.gamma = 0.32

        # State-space matrices
        self.A = np.array([
            [-self.k, -self.gamma, 0, 0],
            [1, 0, 0, 0],
            [0, 1 / self.tau, -1 / (self.alpha * self.tau), 0],
            [0, (self.E0 - (self.E0 - 1) * np.log(1 - self.E0)) / (self.E0 * self.tau),
             (self.alpha - 1) / (self.alpha * self.tau), -1 / self.tau]
        ])

        self.B = np.array([[1], [0], [0], [0]])

        self.C = np.array([[0, 0, self.V0 * (self.k2 - self.k3), self.V0 * (-self.k1 - self.k2)]])

        self.D = np.array([[0]])

    def get_matrices(self):
        """
        Return the state-space matrices A, B, C, D.
        """
        return self.A, self.B, self.C, self.D
