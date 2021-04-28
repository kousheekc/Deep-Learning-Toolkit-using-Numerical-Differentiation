import numpy as np
from deep_learning_toolkit.utils.activation_functions import Softmax


class NeuralNetwork:
    def __init__(self, architecture):
        self.architecture = architecture

        self.parameters = self.initialise_parameters()

    def initialise_parameters(self):
        '''
        Generates the parameters of the neural network based on architecture
        '''
        parameters = []

        for j in range(len(self.architecture)-1):
            parameters.append(np.random.rand(self.architecture[j+1], self.architecture[j]))

        return parameters

    def get_parameters(self):
        '''
        Returns the parameters of the neural network
        '''
        return self.parameters

    def update_parameters(self, new_parameters):
        '''
        Sets new and updated parameters for the model
        '''
        self.parameters = new_parameters

    def forward_propagate(self, parameters, feature_vec_batch):
        '''
        Forward propagates a batch of feature vectors through the neural network
        '''
        all_ai = []

        for feature_vec in feature_vec_batch:
            ai = feature_vec

            for i in range(len(parameters)):
                zi = np.dot(parameters[i], ai)
                ai = Softmax(zi)

            all_ai.append(ai)

        return all_ai

    def predict(self, feature_vec):
        '''
        Predicts the class of one feature vector (used once the neural network is trained)
        '''
        ai = feature_vec

        for i in range(len(self.parameters)):
            zi = np.dot(self.parameters[i], ai)
            ai = Softmax(zi)

        return ai
