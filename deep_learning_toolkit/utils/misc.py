import numpy as np


def tensor_to_vec(parameters):
    parameters_vector = []
    for parameter in parameters:
        parameters_vector = np.concatenate((parameters_vector, parameter.flatten()))

    return parameters_vector

def vec_to_tensor(vector, architecture):
    parameters = []
    count = 0

    for k in range(len(architecture)-1):
        parameter = []
        for _ in range(architecture[k+1]):
            vec = []
            for _ in range(architecture[k]):
                vec.append(vector[count])
                count += 1

            parameter.append(vec)

        parameters.append(np.array(parameter))

    return parameters