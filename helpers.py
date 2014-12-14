import numpy as np
import kernel_helpers as k_helpers
from scipy.optimize import fmin_l_bfgs_b

#fixes vectors to be equal to their expected sums
#(necessary b/c very slight precision errors were screwing
# up the algorithm)
def fix_precision_of_vector(vec, expected_sum):
    u = np.argmax(abs(vec))
    diff = expected_sum - sum(vec)
    vec[u] += diff
    return vec

def get_box_constraints(n, C=1.0):
    C = C * 1.0
    box_constraints = [[0, C] for i in range(0, n)]
    return box_constraints

def compute_J(K, y_mat, alpha0, box_constraints):
    n = K.shape[0]

    def func(alpha):
        """ The SVM dual objective. """
        return (-1 * np.sum(alpha) + 1.0/2 *
                alpha.T.dot(np.multiply(K, y_mat)).dot(alpha))

    def func_deriv(alpha):
        """ Gradient of the SVM dual objective. """
        return -1 * np.ones(n) + np.multiply(K, y_mat).dot(alpha)

    alpha, min_val, info = fmin_l_bfgs_b(func, alpha0, fprime=func_deriv,
                                bounds=box_constraints)

    return alpha, min_val, info

def compute_dJ(kernel_matrices, y_mat, alpha):
    M = len(kernel_matrices)
    n = y_mat.shape[0]
    dJ = np.zeros(M)

    for m in range(M):
        kernel_matrix = kernel_matrices[m]
        dJ[m] = -0.5 * alpha.T.dot(np.multiply(kernel_matrix, y_mat)).dot(alpha)

    return dJ

def get_armijos_step_size(kernel_matrices, d, y_mat, alpha0, box_constraints, gamma0, Jd, D, dJ, c=0.5, T=0.5):
    #m = D' * dJ, should be negative
    #Loop until f(x + gamma * p <= f(x) + gamma*c*m)
    # J(d + gamma * D) <= J(d) + gamma * c * m
    gamma = gamma0
    m = D.T.dot(dJ)

    while True:
        combined_kernel_matrix = k_helpers.get_combined_kernel(kernel_matrices, d + gamma * D)

        alpha, new_J, info = compute_J(combined_kernel_matrix, y_mat, alpha0, box_constraints)

        if new_J <= Jd + gamma * c * m:
            return gamma
        else:
            #Update gamma
            gamma = gamma * T

    return gamma / 2

def compute_descent_direction(d, dJ, mu):
    M = len(d)
    
    #The descent direction
    D = np.zeros(M)

    #Gets descent direction
    for m in range(M):
        #Explained on p. 2498/2499
        if m == mu:
            D[m] = 0
            for v in range(M):
                if (v != mu) and d[v] > 0:
                    D[m] += dJ[v] - dJ[mu]

        #If d[m] == 0, but allow for rounding errors
        elif d[m] > -0.00000001 and d[m] < 0.00000001 and dJ[m] > dJ[mu]:
            #Correct any rounding errors just in case
            d[m] = 0

            #Set descent direction to 0
            D[m] = 0

        elif d[m] > 0 and m != mu:
            D[m] = dJ[mu] - dJ[m]

        else:
            print "fuck up here"
            print d[m]
            print d
            print dJ
            print m
            print mu
            raise Exception('Something went wrong with the descent update!')

    return D


#Returns True if time to stop
def stopping_criterion(dJ, d, threshold):
    M = len(dJ)
    if stopping_criterion.first_iteration:
        stopping_criterion.first_iteration = False
        return False
    else:
        dJ_min = 10000
        dJ_max = -10000
        lowest_dm0 = 100000

        #Gets optimality conditions
        for m in range(M):
            if d[m] > 0:
                if dJ[m] < dJ_min:
                    dJ_min = dJ[m]

                if dJ[m] > dJ_max:
                    dJ_max = dJ[m]
            else:
                if dJ[m] < lowest_dm0:
                    lowest_dm0 = dJ[m]

        return (dJ_max - dJ_min < threshold) and lowest_dm0 >= dJ_max

stopping_criterion.first_iteration = True