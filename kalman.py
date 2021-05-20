'''
Stablize face landmark detection with Kalman filter,
this document support 1D and 2D Kalman filter
Linear
modified from work by yinguobing
'''

import numpy as np
import cv2

import time


class Stabilizer:
    """A kalman filter for 1D or 2D points"""

    def __init__(self, initial_state, input_dim=2,
                 cov_process=0.001, cov_measure=0.01):
        """
        Initialization the stablilizer, 1D for scaler, 2D for 1 point(x y)
        Input: initial_state --- a list of integers to initialize kalman filter
                            1 entry for 1D point, 2 entries for 2D points.
                input_dim --- the dimension of the points for kalman filter. 1 or 2.
                cov_process --- process covariance
                cov_measure --- measure covariance
        """

        # Check: only iput dimension 1 or to allowed
        assert input_dim == 1 or input_dim == 2, "only 1D or 2D allowed"

        # Set up process and input dimensions
        self.state_num = 2 * input_dim
        self.measure_num = input_dim
        self.time_stamp = time.time()

        # initiate filter from opencv
        # No control parameter for now
        self.filter = cv2.KalmanFilter(self.state_num,
                                       self.measure_num,
                                       0)
        # Store the initial state
        if self.measure_num == 1:
            self.state = np.array([[np.float32(initial_state[0])], [0]])
        if self.measure_num == 2:
            self.state = np.array([[np.float32(initial_state[0])],
                                   [np.float32(initial_state[1])],
                                   [0],
                                   [0]])
        # initialize the measurements with 0s
        self.measurement = np.zeros((self.measure_num, 1), np.float32)

        # initialize the prediction results with 0s
        self.prediction = np.zeros((self.state_num, 1), np.float32)

        # Kalman filter parameters setup for 1D
        if self.measure_num == 1:
            self.filter.transitionMatrix = np.array([[1, 1],
                                                     [0, 1]], np.float32)
            self.filter.measurementMatrix = np.array([[1, 1]], np.float32)
            self.filter.processNoiseCov = np.array([[1, 0],
                                                    [0, 1]], np.float32) * cov_process
            self.filter.measurementNoiseCov = np.array([[1]], np.float32) * cov_measure

        # Kalman filter parameters setup for 2D
        if self.measure_num == 2:
            # (x, y, vx, vy) -> x_pred=x+vx, y_pred=y+vy
            self.filter.transitionMatrix = np.array([[1, 0, 1, 0],
                                                     [0, 1, 0, 1],
                                                     [0, 0, 1, 0],
                                                     [0, 0, 0, 1]], np.float32)
            self.filter.measurementMatrix = np.array([[1, 0, 0, 0],
                                                      [0, 1, 0, 0]], np.float32)
            self.filter.processNoiseCov = np.array([[1, 0, 0, 0],
                                                    [0, 1, 0, 0],
                                                    [0, 0, 1, 0],
                                                    [0, 0, 0, 1]], np.float32) * cov_process
            self.filter.measurementNoiseCov = np.array([[1, 0],
                                                        [0, 1]], np.float32) * cov_measure

    def update(self, measurement):
        '''update the kalman filter, containing both prediction by previous results, and the
        correction with new measurements. Results are stored in the self.state.
        Input: measurement --- the new measurement to update kalman filter'''
        # make prediction based on previous results with kalman filter

        dT = time.time() - self.time_stamp
        if self.measure_num == 1:
            self.filter.transitionMatrix[0, 1] = dT
        else:
            self.filter.transitionMatrix[0, 2] = dT
            self.filter.transitionMatrix[1, 3] = dT

        self.prediction = self.filter.predict()

        # Get new measurements
        if self.measure_num == 1:
            self.measurement = np.array([[np.float32(measurement)]])
        else:
            self.measurement = np.array([[np.float32(measurement[0])],
                                         [np.float32(measurement[1])]])

        # correct according to measurement
        self.filter.correct(self.measurement)

        # update the state value
        self.state = self.filter.statePost
        self.time_stamp = time.time()

    def predict(self):
        # make prediction based on previous results with kalman filter
        self.prediction = self.filter.predict()
        self.state = self.prediction
        # return the prediction results
        return self.state

    def get_results(self):
        # get the state of the kalman filter
        if self.state_num == 2:
            return self.state[0]
        if self.state_num == 4:
            return [self.state[0], self.state[1]]


def main():
    """Test code"""
    global mp
    mp = np.array((2, 1), np.float32)  # measurement

    def onmouse(k, x, y, s, p):
        global mp
        mp = np.array([[np.float32(x)], [np.float32(y)]])

    cv2.namedWindow("kalman")
    cv2.setMouseCallback("kalman", onmouse)
    kalman = Stabilizer([0, 0], 2, 1, 1)
    frame = np.zeros((480, 640, 3), np.uint8)  # drawing canvas

    while True:
        kalman.update(mp)
        point = kalman.prediction
        state = kalman.filter.statePost
        result = kalman.get_results()
        cv2.circle(frame, (int(result[0]), int(result[1])), 2, (255, 0, 0), -1)
        cv2.circle(frame, (int(mp[0]), int(mp[1])), 2, (0, 255, 0), -1)
        cv2.imshow("kalman", frame)
        k = cv2.waitKey(30) & 0xFF
        if k == 27:
            break


if __name__ == '__main__':
    main()
