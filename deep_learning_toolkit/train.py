from deep_learning_toolkit.neural_network.neural_network import NeuralNetwork
from deep_learning_toolkit.optimizers.forward_difference_optim import ForwardDifferenceOptim
from deep_learning_toolkit.losses.losses import CrossEntropy
from deep_learning_toolkit.utils.data_preprocessing import convert_classes_for_classification, shuffle_list, make_batches, normalize
from deep_learning_toolkit.utils.logger import Logger
from deep_learning_toolkit.utils.plotter import Plotter
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt


digits_data = datasets.load_digits()

features_vectors = digits_data['data']
labels = digits_data['target']

labels = convert_classes_for_classification(labels)

new_feature_vectors, new_labels = shuffle_list(features_vectors, labels)
new_feature_vectors = np.array(new_feature_vectors)
new_labels = np.array(new_labels)
normalized_feature_vectors = normalize(new_feature_vectors)

feature_vectors_batched, labels_batched = make_batches(5, normalized_feature_vectors, new_labels)

model = NeuralNetwork([64,10])
loss_function = CrossEntropy()
optimizer = ForwardDifferenceOptim(0.05, 1e-5)
logger = Logger('digits', 'forward_difference')
plotter = Plotter('digits', 'forward_difference')

for epoch in range(20):
    print(epoch)
    epoch_loss = 0
    
    for data_batch, label_batch in zip(feature_vectors_batched, labels_batched):
        loss, gradient_tensor = optimizer.compute_gradient(loss_function, model, data_batch, label_batch)

        new_parameters = optimizer.gradient_descent(model, gradient_tensor)

        model.update_parameters(new_parameters)

        epoch_loss += loss

    logger.log(epoch_loss)
    logger.log_model(model.get_parameters())

plotter.plot_individual(True)