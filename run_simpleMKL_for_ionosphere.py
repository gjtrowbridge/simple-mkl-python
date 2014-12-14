import algo1
import numpy as np
import kernel_helpers as k_helpers


data_file = 'ionosphere.data'

data = np.genfromtxt(data_file, delimiter=',', dtype='|S10')
instances = np.array(data[:, :-1], dtype='float')
labels = np.array(data[:, -1] == 'g', dtype='int')

n, d = instances.shape
nlabels = labels.size

if n != nlabels:
    raise Exception('Expected same no. of feature vector as no. of labels')

train_data = instances[:200]  # first 200 examples
train_labels = labels[:200]  # first 200 labels

#Make labels -1 and 1
train_labels[train_labels == 0] = -1

test_data = instances[200:]  # example 201 onwards
test_labels = labels[200:]  # label 201 onwards


# parameters for the kernels we'll use
gamma = 1.0/d
intercept = 0

kernel_functions = [
    k_helpers.linear_kernel,
    k_helpers.create_rbf_kernel(gamma),
    k_helpers.create_poly_kernel(2, gamma),
    k_helpers.create_poly_kernel(3, gamma),
    k_helpers.create_poly_kernel(4, gamma),
    k_helpers.create_sigmoid_kernel(gamma),
]

weights = algo1.find_kernel_weights(train_data, train_labels, kernel_functions)
print 'Final weights for each kernel are:', weights
