import numpy as np
import matplotlib.pyplot as plt
import time


class LinearRegression:

    def __init__(self):
        data_points = 10
        self.data_x = 2 * np.random.rand(data_points, 1)
        noise = np.random.normal(-0.1, 0.1, (data_points, 1))
        self.data_y = 3 * self.data_x + 5 + noise

        self.m = len(self.data_x)
        self.lr = 0.001

        self.slope = -5
        self.y_intercept = 3

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.line1 = None

    def hypothesis(self, x):
        return self.slope*x + self.y_intercept

    def mean_squared_error(self):
        running_total = 0
        for i in range(self.m):
            running_total += (self.hypothesis(self.data_x[i]) - self.data_y[i])**2

        return running_total/(self.m*2)

    def apply_grads(self):

        f = self.data_y - (self.slope * self.data_x + self.y_intercept)
        slope_partial = sum(self.data_x.T.dot(f))/self.m
        y_intercept_partial = (sum(f)/self.m)
        self.slope -= -self.lr * slope_partial
        self.y_intercept -= - self.lr * y_intercept_partial

    def train(self, iterations=10, visualize=True):
        if visualize:
            plt.scatter(self.data_x, self.data_y)
            plt.ion()
            self.line1, = self.ax.plot(self.data_x, self.data_y, 'r-')
        for i in range(iterations):
            self.apply_grads()
            if i % 10 == 0 and visualize:
                self.update_plot()
        print(self.slope)
        print(self.y_intercept)
        print(self.mean_squared_error(), 'loss')

    def update_plot(self):
        self.line1.set_ydata(self.hypothesis(self.data_x))
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        # show the plot
        plt.show()

    def plot(self):
        plt.scatter(self.data_x, self.data_y, label='data')
        plt.plot(self.data_x, self.hypothesis(self.data_x),  'r-', label='hypothesis')
        plt.legend()

        plt.show()


if __name__ == '__main__':
    line = LinearRegression()
    # print(line.mean_squared_error())
    # line.plot()
    line.train(iterations=5000)
    print(line.mean_squared_error())
