import numpy as np

class Amulet:
    def __init__(self, coord_x, coord_y, width, height):
        self.coord_x = coord_x
        self.coord_y = coord_y
        self.width = width
        self.height = height
        print('coordinate:', self.coord_x, self.coord_y, self.width, self.height)

    def compute_labels(self, data, labels):
        self.data = data[self.coord_x:self.coord_x + self.height, self.coord_y:self.coord_y + self.width, :]
        print('cube shape:', self.data.shape)
        self.labels = labels[self.coord_x:self.coord_x + self.height, self.coord_y:self.coord_y + self.width, :]

        self.total_labels = np.sum(self.labels, axis=0)
        self.total_labels = np.sum(self.total_labels, axis=0)

        self.sum_spectrum = np.sum(self.data, axis=0)
        self.sum_spectrum = np.sum(self.sum_spectrum, axis=0)
        print(self.total_labels.shape, self.sum_spectrum.shape)
