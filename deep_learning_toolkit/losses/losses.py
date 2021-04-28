import numpy as np

class MeanSquaredError:
    def __init__(self):
        pass

    def compute_loss(self, prediction_batch, labels_batch):
        loss = (1/(2*len(prediction_batch)))*sum(sum(np.square(np.subtract(prediction_batch, labels_batch))))
        return loss

class CrossEntropy:
    def __init__(self):
        pass

    def compute_loss(self, prediction_batch, labels_batch):
        loss = sum(sum(-np.multiply(labels_batch, prediction_batch)))
        return loss