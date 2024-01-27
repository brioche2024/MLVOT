import numpy
import numpy as np


class KalmanFilter:
    def __init__(self, dt, u_x, u_y, std_acc, x_sdt_meas, y_sdt_meas):
        """
        :param dt: time for one cycle used to estimate state (sampling time)
        :param u_x: accelerations in the x-
        :param u_y: accelerations in the y-
        :param std_acc: process noise magnitude
        :param x_sdt_meas: standard deviations of the measurement in the x-
        :param y_sdt_meas: standard deviations of the measurement in the y-
        """
        self.dt = dt
        self.u_x = u_x
        self.u_y = u_y
        self.std_acc = std_acc
        self.x_sdt_meas = x_sdt_meas
        self.y_sdt_meas = y_sdt_meas

        self.u = np.array([[u_x], [u_y]])
        self.xK = np.zeros((1, 4))
        self.A = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        self.B = np.array([[(dt**2)/2, 0],
                           [0, (dt**2)/2],
                           [dt, 0],
                           [0, dt]])
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        self.Q = np.array([[dt**4/4, 0, dt**3/2, 0],
                           [0, dt**4/4, 0, dt**3/2],
                           [dt**3/2, 0, dt**2, 0],
                           [0, dt**3/2, 0, dt**2]]) * std_acc**2
        self.R = np.array([[x_sdt_meas**2, 0],
                           [0, y_sdt_meas**2]])
        self.P = numpy.identity(4)

    def predict(self):
        # Update state vector xK with the predicted state
        self.xK = np.dot(self.xK, self.A) + np.dot(self.B, self.u)

        # Update error covariance matrix P with the predicted error covariance
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q

        return self.xK, self.P

    def update(self, z):
        # Compute the Kalman Gain
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        # Update the state vector with the new measurement
        z = np.reshape(z, (2, 1))
        y = z - np.dot(self.H, self.xK)
        self.xK = self.xK + np.dot(K, y)

        # Update the error covariance matrix
        I = np.eye(self.A.shape[0])
        self.P = np.dot((I - np.dot(K, self.H)), self.P)

        return self.xK, self.P
