from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
from sklearn import tree
import copy
from sklearn.decomposition import PCA
from utils import *


#converts list to string
def convert_to_str(list):
     string = ''.join(str(number) for number in list)
     return string

# attempt at feature hashing
def hashing_function(subject_list, length ):
    hashed_list =[0]*length
    for subject_id in subject_list:
        h = hash(int(subject_id))
        hashed_list[h % length] += 1

    hash_string = convert_to_str(hashed_list)
    return hash_string


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

    for i in range(iterations):
        neg_lld = neg_log_likelihood(matrix, theta=theta, beta=beta)
        score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(score)
        print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta = update_theta_beta(matrix, lr, theta, beta)

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, val_acc_lst


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


def create_data(data, theta,beta,question_data):
    user_ids = data.get("user_id")
    data["subject"] = []
    counter = 1
    for i in range(0,388):
        data[str(i)] = []
        
    new_list_one = []
    question_ids = data.get("question_id")
    new_list_two = []
    for user_id in user_ids:
        new_list_one.append(theta[user_id])
    for question_id in question_ids:
        new_list_two.append(beta[question_id])
        subject_string = question_data.get(question_id)
        subject_list = subject_string[1:len(subject_string)-1].split(",")
        data.get("subject").append(hashing_function(subject_list,10))
        for i in range(0,388):
            if i in list(map(int, subject_list)):
                data.get(str(i)).append(1)
            else:
                data.get(str(i)).append(0)

    data["user_id"] = new_list_one
    data["question_id"] = new_list_two
    df = pd.DataFrame.from_dict(data)
    return df,data

#logistic regression
def run_logistic_regression(train_data,sparse_matrix, test_data, question_data,theta,beta,k):
    df_train, train_data = create_data(train_data,theta,beta,question_data)
    df_test, test_data = create_data(test_data,theta,beta,question_data)
    X_train  = df_train.drop(["is_correct","subject"], axis=1)
    pca = PCA(n_components = k,random_state = 42)
    X_train =pca.fit_transform(X_train)
    y_train = df_train["is_correct"]
    X_test = df_test.drop(["is_correct","subject"], axis=1)
    y_test = df_test["is_correct"]
    X_test =pca.transform(X_test)
    logistic_model = LogisticRegression(max_iter = 1000)
    logistic_model.fit(X_train,y_train)
    y_pred = logistic_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

#decision tree
def run_decision_tree(train_data,sparse_matrix, test_data, question_data,theta,beta,k):
    df_train, train_data = create_data(train_data,theta,beta,question_data)
    df_test, test_data = create_data(test_data,theta,beta,question_data)
    X_train  = df_train.drop(["is_correct","subject"], axis=1)
    pca = PCA(n_components = k,random_state = 42)
    X_train =pca.fit_transform(X_train)
    y_train = df_train["is_correct"]
    X_test = df_test.drop(["is_correct","subject"], axis=1)
    y_test = df_test["is_correct"]
    X_test =pca.transform(X_test)
    decision_tree = tree.DecisionTreeClassifier()
    decision_tree = decision_tree.fit(X_train,y_train)
    y_pred = decision_tree.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# tuning k value for pca decomposition
def tune_decompostion(train_data,sparse_matrix, test_data, question_data,theta,beta):
    acc_lst_log = []
    acc_lst_dec = []
    k_list = [5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]
    for k in k_list:
        logistic_regression_accuracy = run_logistic_regression(copy.deepcopy(train_data),sparse_matrix,copy.deepcopy(test_data),question_data,theta,beta,k)
        decision_tree_accuracy = run_decision_tree(copy.deepcopy(train_data),sparse_matrix,copy.deepcopy(test_data),question_data,theta,beta,k)
        acc_lst_log.append(logistic_regression_accuracy)
        acc_lst_dec.append(decision_tree_accuracy)

    return acc_lst_log,acc_lst_dec



def main():
    train_data = load_train_csv("./data")
    sparse_matrix = load_train_sparse("./data").toarray()
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")
    question_data = load_question_csv("./data")
   # accuracy = cross_val_score(logistic_model,X,y_true,scoring = "accuracy").mean()
    #logistic_regression_accuracy = run_logistic_regression(copy.deepcopy(train_data),sparse_matrix,copy.deepcopy(test_data),question_data,5)
   #decision_tree_accuracy = run_decision_tree(copy.deepcopy(train_data),sparse_matrix,copy.deepcopy(test_data),question_data,5)

   #using all tuned values when running prediction models
    theta , beta, acc_lst = irt(sparse_matrix, val_data, 0.01, 200)
    train_accuracy_log = run_logistic_regression(copy.deepcopy(train_data),sparse_matrix,copy.deepcopy(train_data),question_data,theta,beta,15)
    val_accuracy_log = run_logistic_regression(copy.deepcopy(train_data),sparse_matrix,copy.deepcopy(test_data),question_data,theta,beta,15)
    test_accuracy_log = run_logistic_regression(copy.deepcopy(train_data),sparse_matrix,copy.deepcopy(test_data),question_data,theta,beta,15)
    train_accuracy_dec = run_decision_tree(copy.deepcopy(train_data),sparse_matrix,copy.deepcopy(train_data),question_data,theta,beta,2)
    val_accuracy_dec = run_decision_tree(copy.deepcopy(train_data),sparse_matrix,copy.deepcopy(test_data),question_data,theta,beta,2)
    test_accuracy_dec = run_decision_tree(copy.deepcopy(train_data),sparse_matrix,copy.deepcopy(test_data),question_data,theta,beta,2)
    print("Logistic Regression Training Accuracy: "+ str(train_accuracy_log))
    print("Logistic Regression Validation Accuracy: "+ str(val_accuracy_log))
    print("Logistic Regression Testing Accuracy: "+ str(test_accuracy_log))
    print("Decision Tree Training Accuracy: "+ str(train_accuracy_dec))
    print("Decision Tree Validation Accuracy: "+ str(val_accuracy_dec))
    print("Decision Tree Testing Accuracy: "+ str(test_accuracy_dec))
    #acc_lst_log , acc_lst_dec = tune_decompostion(train_data,sparse_matrix,test_data,question_data,theta,beta)
    #print(max(acc_lst_log))
    #print(max(acc_lst_dec))
    #k_list = [5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]
    #plt.scatter(k_list, acc_lst_log)
    #plt.xlabel("k value")
    #plt.ylabel("Logistic Regression Accuracy")
    #plt.ylim(0.65, 0.75)
    #plt.savefig('plot_one.png')
    #plt.show()
    #plt.scatter(k_list, acc_lst_dec)
    #plt.xlabel("k value")
    #plt.ylabel("Decision Tree Accuracy")
    #plt.ylim(0.55, 0.68)
    #plt.savefig('plot_two.png')
    #plt.show()
   #print(logistic_regression_accuracy)
    #print(decision_tree_accuracy)

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()

