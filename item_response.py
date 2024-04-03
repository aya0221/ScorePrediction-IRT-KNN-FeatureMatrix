from utils import *
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import random


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))

def get_diff_matrix(matrix,theta,beta):
    diff_matrix = np.array([[theta[i] - beta[j] for j in range(0,matrix.shape[1])] for i in range(0,matrix.shape[0])])
    return diff_matrix

#def convert_to_sparse(data):


def neg_log_likelihood(matrix, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    diff_matrix = get_diff_matrix(matrix,theta,beta)
    lklihood_matrix = np.multiply(matrix,diff_matrix) - np.log(1+np.exp(diff_matrix))
    lklihood_matrix[np.isnan(matrix)] = 0
    log_lklihood = np.sum(lklihood_matrix)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood



def update_theta_beta(matrix, lr, theta, beta):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    diff_matrix = get_diff_matrix(matrix,theta,beta)
    final_matrix = matrix - sigmoid(diff_matrix)
    final_matrix[np.isnan(final_matrix)] = 0
    deriv_theta = np.sum(final_matrix,axis = 1)
    theta += (lr*deriv_theta)
    diff_matrix = get_diff_matrix(matrix,theta,beta)
    final_matrix =  sigmoid(diff_matrix) - matrix
    final_matrix[np.isnan(final_matrix)] = 0
    deriv_beta = np.sum(final_matrix,axis = 0)
    beta += (lr*deriv_beta)

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt(matrix, val_data, lr, iterations):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # TODO: Initialnumpy.random.randize theta and beta.
    theta = np.random.rand(matrix.shape[0])
    beta = np.random.rand(matrix.shape[1])
    val_acc_lst = []
    lklihood_list = []

    for i in range(iterations):
        neg_lld = neg_log_likelihood(matrix, theta=theta, beta=beta)
        lklihood_list.append(neg_lld*-1)
        score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(score)
        print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta = update_theta_beta(matrix, lr, theta, beta)

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, val_acc_lst,lklihood_list


def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])

#tuning hyperparameters
def model_tuning(matrix, val_data):
    best_accuracy = -1
    optimal_iter = 100
    optimal_lr = 0.001
    iter_list = [100,200,500,1000]
    lr_list = [0.001, 0.01, 0.05, 0.1]
    for iteration in iter_list:
        for lr in lr_list:
            theta , beta, val_acc_lst, lklihood_list= irt(matrix, val_data, lr, iteration)
            if val_acc_lst[len(val_acc_lst)-1] > best_accuracy:
                best_accuracy = val_acc_lst[len(val_acc_lst)-1]
                optimal_iter = iteration
                optimal_lr = lr
    return optimal_lr, optimal_iter

# plotting questions - part d
def plot_data(id_one, id_two, id_three, theta,beta):
    probability_list_one = []
    probability_list_two= []
    probability_list_three = []
    for value in theta:
        probability_list_one.append(sigmoid(value - beta[id_one]))
    for value in theta:
        probability_list_two.append(sigmoid(value - beta[id_two]))
    for value in theta:
        probability_list_three.append(sigmoid(value - beta[id_three]))
    plt.plot(theta,probability_list_one, label='Question 1')
    plt.plot(theta,probability_list_two, label='Question 2')
    plt.plot(theta,probability_list_three, label='Question 3')
    plt.legend()
    plt.xlabel('Theta')
    plt.ylabel('Probability of Correct Response')
    plt.grid(True)
    plt.show()

def main():
    train_data = load_train_csv("./data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("./data").toarray()
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")
    #optimal_lr, optimal_iter = model_tuning(sparse_matrix,val_data)
    #print("Optimal Learning Rate: "+str(optimal_lr))
    #print("Optimal Iteration Number: "+str(optimal_iter))
    print()
    print("------Validation Accuracies------")
    print()
    theta , beta, val_acc_lst,lklihood_list = irt(sparse_matrix, val_data, 0.01, 200)
    plt.plot(range(1,len(lklihood_list)+1), lklihood_list)
    plt.xlabel("Iteration #")
    plt.ylabel("Training Log Likelihood")
    plt.grid(True)
    plt.show()
    print()
    print("------Test Accuracies------")
    print()
    theta , beta, test_acc_lst,lklihood_list = irt(sparse_matrix, test_data, 0.01, 200)
    q1 = random.randint(0,len(beta)-1)
    q2 = random.randint(0,len(beta)-1)
    q3 = random.randint(0,len(beta)-1)
    plot_data(q1,q2,q3,np.sort(theta),beta)
   # print(evaluate(train_data,theta,beta))
    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    pass
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (d)                                                #
    #####################################################################
    pass
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
