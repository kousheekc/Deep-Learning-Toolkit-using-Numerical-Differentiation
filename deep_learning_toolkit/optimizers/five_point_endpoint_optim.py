import numpy as np
from deep_learning_toolkit.utils.misc import tensor_to_vec, vec_to_tensor


class FivePointEndpointOptim:
    def __init__(self, lr, epsilon):
        self.lr = lr
        self.epsilon = epsilon

    def compute_gradient(self, loss_function, model, data_batch, label_batch):
        gradient_vector = []

        predictions_batch = model.forward_propagate(model.get_parameters(), data_batch)
        loss = loss_function.compute_loss(predictions_batch, label_batch)

        parameter_vector_plus_epsilon = tensor_to_vec(model.get_parameters())
        parameter_vector_plus_two_epsilon = tensor_to_vec(model.get_parameters())
        parameter_vector_plus_three_epsilon = tensor_to_vec(model.get_parameters())
        parameter_vector_plus_four_epsilon = tensor_to_vec(model.get_parameters())

        for i in range(len(parameter_vector_plus_epsilon)):
            parameter_vector_plus_epsilon[i] += self.epsilon
            parameter_vector_plus_two_epsilon[i] += 2*self.epsilon
            parameter_vector_plus_three_epsilon[i] += 3*self.epsilon
            parameter_vector_plus_four_epsilon[i] += 4*self.epsilon

            new_parameters_plus_epsilon = vec_to_tensor(parameter_vector_plus_epsilon, model.architecture)
            new_parameters_plus_two_epsilon = vec_to_tensor(parameter_vector_plus_two_epsilon, model.architecture)
            new_parameters_plus_three_epsilon = vec_to_tensor(parameter_vector_plus_three_epsilon, model.architecture)
            new_parameters_plus_cour_epsilon = vec_to_tensor(parameter_vector_plus_four_epsilon, model.architecture)

            new_predictions_plus_epsilon = model.forward_propagate(new_parameters_plus_epsilon, data_batch)
            new_predictions_plus_two_epsilon = model.forward_propagate(new_parameters_plus_two_epsilon, data_batch)
            new_predictions_plus_three_epsilon = model.forward_propagate(new_parameters_plus_three_epsilon, data_batch)
            new_predictions_plus_four_epsilon = model.forward_propagate(new_parameters_plus_cour_epsilon, data_batch)

            new_loss_plus_epsilon = loss_function.compute_loss(new_predictions_plus_epsilon, label_batch)
            new_loss_plus_two_epsilon = loss_function.compute_loss(new_predictions_plus_two_epsilon, label_batch)
            new_loss_plus_three_epsilon = loss_function.compute_loss(new_predictions_plus_three_epsilon, label_batch)
            new_loss_plus_four_epsilon = loss_function.compute_loss(new_predictions_plus_four_epsilon, label_batch)

            dLoss_dParameter_i = (-25*loss + 48*new_loss_plus_epsilon - 36*new_loss_plus_two_epsilon + 16*new_loss_plus_three_epsilon - 3*new_loss_plus_four_epsilon)/(12*self.epsilon)

            gradient_vector.append(dLoss_dParameter_i)

        dLoss_dParameters = vec_to_tensor(gradient_vector, model.architecture)

        return loss, dLoss_dParameters

    def gradient_descent(self, model, gradient_tensor):
        new_parameters = []

        for i, parameter_matrix in enumerate(model.get_parameters()):
            new_parameters.append(np.subtract(parameter_matrix, self.lr * gradient_tensor[i]))

        return new_parameters

    