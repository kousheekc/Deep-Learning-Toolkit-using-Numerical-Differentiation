import numpy as np
from random import shuffle

def shuffle_list(*ls):
    l =list(zip(*ls))

    shuffle(l)
    return zip(*l)

def convert_classes_for_classification(classes):
    new_classes = []
    for c in classes:
        new_class = [0]*(max(classes)+1)
        new_class[int(c)] += 1
        new_classes.append(new_class)

    return np.array(new_classes)

def make_batches(batch_size, feature_vectors, labels):
    feature_vectors_batches = []
    labels_batches = []

    for i in range(0, len(feature_vectors), batch_size):
        feature_vectors_batches.append(feature_vectors[i:i+batch_size])
        labels_batches.append(labels[i:i+batch_size])

    return feature_vectors_batches, labels_batches

def normalize(feature_vectors):
    return np.divide(feature_vectors, np.max(feature_vectors))