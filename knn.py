from sklearn.impute import KNNImputer
from utils import *

def knn_impute_by_user(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.
    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.
    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k) # completing missing values using k-Nearest Neighbors / returns a scalar distance value.
    # We use NaN-Euclidean distance measure.(default) 
    mat = nbrs.fit_transform(matrix) 
    acc = sparse_matrix_evaluate(valid_data, mat) 
    print("Validation Accuracy: {}".format(acc))
    return acc


def knn_impute_by_item(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.
    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    # sparse_matrix.npz: 
      # - rows: user_id
      # - columns: question_id
    matrix_T = matrix.transpose()
    nbrs = KNNImputer(n_neighbors=k) # completing missing values using k-Nearest Neighbors / returns a scalar distance value.
    # We use NaN-Euclidean distance measure.(defaut) 
    mat = nbrs.fit_transform(matrix_T) 
    # Transpose back
    mat_T = mat.transpose() 
    acc = sparse_matrix_evaluate(valid_data, mat_T) 
    print("Validation Accuracy: {}".format(acc))
    return acc
    # acc = None
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return acc



def main():
    sparse_matrix = load_train_sparse("./data").toarray()
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)

    #####################################################################
    # TODO:                                                             #
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*.                                                        #
    #####################################################################

    print('\n=====Answer (a)=====')
    # runs kNN for different values and plot
    k = [1, 6, 11, 16, 21, 26]
    val_acc = []
    
    for i in range(len(k)):
      val_acc.append(knn_impute_by_user(sparse_matrix, val_data, k[i]))
    
    plt.plot(k, val_acc)
    plt.xlabel("K")
    plt.ylabel("Validation Accuracy")
    plt.show()
    

    print('\n=====Answer (b)=====')
    # Choose 𝑘 that has the highest performance on validation data. Report the chosen 𝑘 and the final test accuracy.
    opt_k = k[val_acc.index(max(val_acc))]
    print('𝑘 that has the highest performance on validation data: {}'.format(opt_k))

    test_acc = knn_impute_by_user(sparse_matrix, test_data, 11)
    print('The final test accuracy: {}'.format(test_acc))


    print('\n=====Answer (c)=====')
    # runs kNN for different values and plot
    k = [1, 6, 11, 16, 21, 26]
    val_acc_item = []
    
    for i in range(len(k)):
      val_acc_item.append(knn_impute_by_item(sparse_matrix, val_data, k[i]))
    
    plt.plot(k, val_acc_item)
    plt.xlabel("K")
    plt.ylabel("Validation Accuracy (item-based)")
    plt.show()
    
    # Choose 𝑘 that has the highest performance on validation data. Report the chosen 𝑘 and the final test accuracy.
    opt_k_item = k[val_acc_item.index(max(val_acc_item))]
    print('𝑘 that has the highest performance on validation data (item-based): {}'.format(opt_k_item))

    test_acc_item = knn_impute_by_item(sparse_matrix, test_data, 11)
    print('The final test accuracy (item-based): {}'.format(test_acc_item))


    print('\n=====Answer (d)=====')
    if test_acc < test_acc_item:
      print('item-based filtering performs better than user-based for {}.'.format(test_acc_item-test_acc))
    else:
      print('user-based filtering performs better than item-based for {}.'.format(test_acc-test_acc_item))

    # pass
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()



'''please see the report
(e) List at least two potential limitations of kNN for the task you are given.
- kNN could perform slowly and be computationally intensive, especially when our datasets are large, assumi g that online materials will easily be a huge amount of data collected.
- It is sometimes difficult to grasp the reason why the model is making certain predictions as kNN is not a parametric model thought kNN could capture complex patterns of data that are challenging to model with other parametric methods.
- Picking not the best k will easily cause poor predictions; when k is too small, the underlying patterns in the data might not be able to be captured by the model, and if k is too large, the model may have a chance to be overly influenced by outliers or noise in the data.
'''