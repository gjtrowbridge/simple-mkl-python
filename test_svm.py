import algo1
import numpy as np
import kernel_helpers as k_helpers


def svm_classifier(point):
    pass
    # #Returns 1 for class positive, 0 for class negative
    # calc_kernel = lambda x_i: kernel_func(point, x_i)
    # kernel_vector = np.apply_along_axis(calc_kernel, axis=1,
    #                                     arr=support_vectors)
    # value = np.inner(alpha_y_nz, kernel_vector)
    # if value > 0:
    #     return 1
    # else:
    #     return 0
    # return classifier


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
    k_helpers.create_poly_kernel(3, gamma),
    k_helpers.create_poly_kernel(4, gamma),
    k_helpers.create_poly_kernel(5, gamma),
    k_helpers.create_poly_kernel(2, gamma),
    k_helpers.linear_kernel,
    k_helpers.create_rbf_kernel(gamma),
    k_helpers.create_sigmoid_kernel(gamma),
]

print 'yo'
weights = algo1.find_kernel_weights(train_data, train_labels, kernel_functions)
print weights
print sum(weights)
