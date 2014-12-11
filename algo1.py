import numpy as np
import helpers
import kernel_helpers as k_helpers
from sklearn import svm

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

def find_kernel_weights(X, y, kernel_functions):
    # The number of kernels
    M = len(kernel_functions)

    #The number of examples
    n = len(y)

    #The weights of each kernel
    #Initialized to 1/M
    d = np.ones(M) / M

    #Just a placeholder for something that gets updated later
    dJ = '-20'

    #Stores all the individual kernel matrices
    kernel_matrices = k_helpers.get_all_kernels(X, kernel_functions)

    #Creates y matrix for use in SVM later
    #y's should be -1, 1 before doing this.
    y_mat = np.outer(y, y)

    #Gets constraints for running SVM
    box_constraints = helpers.get_box_constraints(n, 1.0)

    #Gets starting value for SVM
    alpha0 = np.zeros(n)

    iteration = 0

    #Loops until stopping criterion reached
    while (iteration < 15):#not helpers.stopping_criterion(dJ, d, 0.01)):
        iteration += 1
        print "iteration and weights:"
        print iteration
        print d

        combined_kernel_matrix = k_helpers.get_combined_kernel(kernel_matrices, d)
        combined_kernel_func = k_helpers.get_combined_kernel_function(kernel_functions, d)
        print is_pos_def(combined_kernel_matrix)
        
        clf = svm.SVC(kernel=combined_kernel_func)
        clf.fit(X, y)
        alpha = np.zeros(n)
        for svi in range(len(clf.support_))
            

            alpha[i] = 

        #Gets J, also calculates the optimal values for alpha
        alpha, J, info = helpers.compute_J(combined_kernel_matrix, y_mat, alpha0, box_constraints)
        print '~~~~~~~~~~~~~'
        print len(clf.support_)
        print len(clf.dual_coef_[0])
        print '~~~~~~~~~~~~~'

        print alpha
        print '~~~~~~~~~~~~~'
        J *= -1

        #Gradient of J w.r.t d (weights)
        dJ = helpers.compute_dJ(kernel_matrices, y_mat, alpha)
        
        #The index of the largest component of d
        mu = d.argmax()

        #Descent direction
        #Basically, we are calculating -1 * reduced gradient of J w.r.t d,
        #using the index of the largest component of d as our "mu"
        #in the reduced gradient calculation
        D = helpers.compute_descent_direction(d, dJ, mu)
        print dJ
        print d
        print D
        J_cross = 0
        d_cross = d.copy()
        D_cross = D.copy()

        sub_iteration = 0

        #Get maximum admissible step size in direction D
        while (J_cross < J):
            sub_iteration += 1
            d = d_cross.copy()
            D = D_cross.copy()

            #Maximum admissible step size
            gamma_max = 123456

            #argmax of above
            v = -0.123456

            #Find gamma_max and v
            for m in range(M):
                if D[m] < 0:
                    d_D_quotient = -1 * d[m] / D[m]
                    if d_D_quotient < gamma_max:
                        gamma_max = d_D_quotient
                        v = m

            d_cross = d + gamma_max * D
            print '^^^^^^^'
            print y.T.dot(alpha)
            print gamma_max
            print '^^^^^^^'

            print d_cross
            print v
            print '******'

            print D_cross
            print '******'
            D_cross[mu] = D[mu] - D[v]
            D_cross[v] = 0

            print D_cross
            # D_cross = D_cross / np.linalg.norm(D_cross)

            combined_kernel_matrix = k_helpers.get_combined_kernel(kernel_matrices, d_cross)
            alpha_cross, J_cross, cross_info = helpers.compute_J(combined_kernel_matrix, y_mat, alpha, box_constraints)
            J_cross *= -1
            print '-------'
            print J
            print J_cross
            print '-------'
            if sub_iteration > 5:
                raise Exception('asdg')
            # J_cross = J + 1

        #Line search along D for gamma (step) in [0, gamma_max]
        # gamma = helpers.get_armijos_step_size()
        gamma = helpers.get_armijos_step_size(kernel_matrices, d, y_mat, alpha,
                                              box_constraints, gamma_max, J_cross,
                                              D, dJ)

        d += gamma * D

    #Return final weights
    return d

# X = np.ones((5,5)) + (np.eye(5) * 2)
# X[2, 1] = 3
# X[3, 1] = 2.5
# y = np.ones((5,1))
# y[2] = -1
# y[3] = -1
# kernel_functions = [k_helpers.create_poly_kernel(degree=3, gamma=0.25),
#                     k_helpers.linear_kernel,
#                     k_helpers.create_poly_kernel(degree=5, gamma=0.3)]

# d = find_kernel_weights(X, y, kernel_functions)
