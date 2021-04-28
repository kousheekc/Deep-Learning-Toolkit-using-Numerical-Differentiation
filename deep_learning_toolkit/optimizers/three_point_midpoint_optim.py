import numpy as np
from deep_learning_toolkit.utils.misc import tensor_to_vec, vec_to_tensor


class ThreePointMidpointOptim:
    def __init__(self, lr, epsilon):
        self.lr = lr
        self.epsilon = epsilon

    def compute_gradient(self, loss_function, model, data_batch, label_batch):
        gradient_vector = []

        predictions_batch = model.forward_propagate(model.get_parameters(), data_batch)
        loss = loss_function.compute_loss(predictions_batch, label_batch)

        parameter_vector_plus_epsilon = tensor_to_vec(model.get_parameters())
        parameter_vector_minus_epsilon = tensor_to_vec(model.get_parameters())

        for i in range(len(parameter_vector_plus_epsilon)):
            parameter_vector_plus_epsilon[i] += self.epsilon
            parameter_vector_minus_epsilon[i] -= self.epsilon

            new_parameters_plus_epsilon = vec_to_tensor(parameter_vector_plus_epsilon, model.architecture)
            new_parameters_minus_epsilon = vec_to_tensor(parameter_vector_minus_epsilon, model.architecture)

            new_predictions_plus_epsilon = model.forward_propagate(new_parameters_plus_epsilon, data_batch)
            new_predictions_minus_epsilon = model.forward_propagate(new_parameters_minus_epsilon, data_batch)

            new_loss_plus_epsilon = loss_function.compute_loss(new_predictions_plus_epsilon, label_batch)
            new_loss_minus_epsilon = loss_function.compute_loss(new_predictions_minus_epsilon, label_batch)

            dLoss_dParameter_i = (new_loss_plus_epsilon - new_loss_minus_epsilon)/(2*self.epsilon)

            gradient_vector.append(dLoss_dParameter_i)

        dLoss_dParameters = vec_to_tensor(gradient_vector, model.architecture)

        return loss, dLoss_dParameters

    def gradient_descent(self, model, gradient_tensor):
        new_parameters = []

        for i, parameter_matrix in enumerate(model.get_parameters()):
            new_parameters.append(np.subtract(parameter_matrix, self.lr * gradient_tensor[i]))

        return new_parameters

    