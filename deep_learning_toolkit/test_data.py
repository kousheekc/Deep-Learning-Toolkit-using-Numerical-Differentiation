import numpy as np
from sklearn import datasets
from deep_learning_toolkit.utils.data_preprocessing import convert_classes_for_classification, shuffle_list, make_batches, normalize

digits_data = datasets.load_digits()

features_vectors = digits_data['data']
labels = digits_data['target']

labels = convert_classes_for_classification(labels)

new_feature_vectors, new_labels = shuffle_list(features_vectors, labels)
new_feature_vectors = np.array(new_feature_vectors)
new_labels = np.array(new_labels)
normalized_feature_vectors = normalize(new_feature_vectors)

feature_vectors_batched, labels_batched = make_batches(64, normalized_feature_vectors, new_labels)

print(feature_vectors_batched[0][0])
