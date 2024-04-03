from utils import *
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

import numpy as np
import torch
import matplotlib.pyplot as plt
#test

def load_data(base_path="./data"):
    """ Load the data in PyTorch Tensor.

    :return: (zero_train_matrix, train_data, valid_data, test_data)
        WHERE:
        zero_train_matrix: 2D sparse matrix where missing entries are
        filled with 0.
        train_data: 2D sparse matrix
        valid_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
        test_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
    """
    train_matrix = load_train_sparse(base_path).toarray()
    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)

    zero_train_matrix = train_matrix.copy()
    # Fill in the missing entries to 0.
    zero_train_matrix[np.isnan(train_matrix)] = 0
    # Change to Float Tensor for PyTorch.
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)

    return zero_train_matrix, train_matrix, valid_data, test_data


 
class AutoEncoder(nn.Module):
    def __init__(self, num_question, k=100):
        """ Initialize a class AutoEncoder.

        :param num_question: int
        :param k: int
        """
        super(AutoEncoder, self).__init__()

        # Define linear functions. (used for forward pass)
        self.g = nn.Linear(num_question, k)
        self.h = nn.Linear(k, num_question)

    def get_weight_norm(self):
        """ Return ||W^1||^2 + ||W^2||^2.

        :return: float
        """
        g_w_norm = torch.norm(self.g.weight, 2) ** 2 # get norm of the weight g matrix
        h_w_norm = torch.norm(self.h.weight, 2) ** 2
        return g_w_norm + h_w_norm

    def forward(self, inputs):
        """ Return a forward pass given inputs.

        :param inputs: user vector.
        :return: user vector.
        """
        #####################################################################
        # TODO:                                                             #
        # Implement the function as described in the docstring.             #
        # Use sigmoid activations for f and g.                              #
        #####################################################################
        out = inputs
        # (a)
        out = F.sigmoid(self.g(out))
        out = F.sigmoid(self.h(out))
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
        return out

class AutoEncoder_l2(nn.Module): # (d) add l2 regularization
    def __init__(self, num_question, k=100):
        """ Initialize a class AutoEncoder.

        :param num_question: int
        :param k: int
        """
        super(AutoEncoder_l2, self).__init__()

        # Define linear functions. (used for forward pass)
        self.g = nn.Linear(num_question, k)
        self.h = nn.Linear(k, num_question)

    def get_weight_norm(self,lamda):
        """ Return ||W^1||^2 + ||W^2||^2 + lambda_l2 * (||W^1||^2 + ||W^2||^2) <= L2
        :return: float
        """sss
        g_w_norm = torch.norm(self.g.weight, 2) ** 2 # get norm of the weight g matrix
        h_w_norm = torch.norm(self.h.weight, 2) ** 2
        return (lamda/2 * (g_w_norm + h_w_norm))

    def forward(self, inputs):
        """ Return a forward pass given inputs.

        :param inputs: user vector.
        :return: user vector.
        """
        #####################################################################
        # TODO:                                                             #
        # Implement the function as described in the docstring.             #
        # Use sigmoid activations for f and g.                              #
        #####################################################################
        out = inputs
        # (a)
        out = F.sigmoid(self.g(out))
        out = F.sigmoid(self.h(out))
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
        return out

def train(model, lr, lamb, train_data, zero_train_data, valid_data, num_epoch):
    """ Train the neural network, where the objective also includes
    a regularizer.

    :param model: Module
    :param lr: float
    :param lamb: float
    :param train_data: 2D FloatTensor
    :param zero_train_data: 2D FloatTensor
    :param valid_data: Dict
    :param num_epoch: int
    :return: None
    """
    # TODO: Add a regularizer to the cost function. 
    
    # Tell PyTorch you are training the model.
    model.train()

    # Define optimizers and loss function.
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_student = train_data.shape[0]

    val_accs = []
    valid_acc = 0

    for epoch in range(0, num_epoch):
        train_loss = 0.

        for user_id in range(num_student):
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0)
            target = inputs.clone()

            optimizer.zero_grad()
            output = model(inputs)

            # Mask the target to only compute the gradient of valid entries.
            nan_mask = np.isnan(train_data[user_id].unsqueeze(0).numpy())
            target[0][nan_mask] = output[0][nan_mask]

            loss = torch.sum((output - target) ** 2.) 
            loss.backward()

            train_loss += loss.item()
            optimizer.step() #Performs a single optimization step (parameter update) (b)

        valid_acc = evaluate(model, zero_train_data, valid_data)
        val_accs.append(valid_acc)
        print("Epoch: {} \tTraining Cost: {:.6f}\t "
              "Valid Acc: {}".format(epoch, train_loss, valid_acc))
        
    return valid_acc, val_accs

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def train_l2(model, lr, lamb, train_data, zero_train_data, valid_data, num_epoch, lambda_l2):
    """ Train the neural network, where the objective also includes
    a regularizer.

    :param model: Module
    :param lr: float
    :param lamb: float
    :param train_data: 2D FloatTensor
    :param zero_train_data: 2D FloatTensor
    :param valid_data: Dict
    :param num_epoch: int
    :return: None
    """
    # TODO: Add a regularizer to the cost function. 
    
    # Tell PyTorch you are training the model.
    model.train()

    # Define optimizers and loss function.
    optimizer = optim.SGD(model.parameters(), lr=lr)
    # optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=lambda_l2)
    num_student = train_data.shape[0]

    val_accs = []
    valid_acc = 0

    for epoch in range(0, num_epoch):
        train_loss = 0.

        for user_id in range(num_student):
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0)
            target = inputs.clone()

            optimizer.zero_grad()
            output = model(inputs)

            # Mask the target to only compute the gradient of valid entries.
            nan_mask = np.isnan(train_data[user_id].unsqueeze(0).numpy())
            target[0][nan_mask] = output[0][nan_mask]

            loss = torch.sum((output - target) ** 2.) + model.get_weight_norm(lambda_l2)
            loss.backward()

            train_loss += loss.item()
            optimizer.step() #Performs a single optimization step (parameter update) (b)

        valid_acc = evaluate(model, zero_train_data, valid_data)
        val_accs.append(valid_acc)
        print("Epoch: {} \tTraining Cost: {:.6f}\t "
              "Valid Acc: {}".format(epoch, train_loss, valid_acc))
        
    return valid_acc, val_accs

def evaluate(model, train_data, valid_data):
    """ Evaluate the valid_data on the current model.

    :param model: Module
    :param train_data: 2D FloatTensor
    :param valid_data: A dictionary {user_id: list,
    question_id: list, is_correct: list}
    :return: float
    """
    # Tell PyTorch you are evaluating the model.
    model.eval()

    total = 0
    correct = 0

    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        output = model(inputs)

        guess = output[0][valid_data["question_id"][i]].item() >= 0.5
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1
    return correct / float(total)


def main():
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()

    #####################################################################
    # TODO:    (b)                                                         #
    # Try out 5 different k and select the best k using the             #
    # validation set.                                                   #
    #####################################################################
    # Set model hyperparameters.
    print('=== (b) ===')
    k_lists =  [10, 50, 100, 200, 500]
    # model = None

    # Set optimization hyperparameters.
    lr = 0.5
    num_epoch = 5
    lamb = 1.5

    num_question = train_matrix.shape[1] # + len(test_data)
    # acc_list = []
    # for k in k_lists:
    #   print('\n k = {}\n'.format(k))
    #   model = AutoEncoder(num_question=num_question, k=k)
    #   val_acc = (train(model, lr, lamb, train_matrix, zero_train_matrix,
    #         valid_data, num_epoch))[0] # return each accuracy
    #   acc_list.append(val_acc)

    # opt_k_item = k_lists[acc_list.index(max(acc_list))]
    # print('𝑘∗ that has the highest validation accuracy: {}\n'.format(opt_k_item)) 

    # (c)
    opt_k_item = 500
    print('=== (c) ===')
    model_epoc = AutoEncoder(num_question=num_question, k=opt_k_item)
    val_accs = (train(model_epoc, lr, lamb, train_matrix, zero_train_matrix,
            valid_data, num_epoch))[1] #return list
    
    plt.plot(range(num_epoch), val_accs)
    plt.xlabel('Epoch')
    plt.ylabel('Validation accuracy')
    plt.legend()
    # plt.show()

    test_accs = evaluate(model_epoc, zero_train_matrix, test_data) 
    print('Final accuracy: {}'.format(test_accs))
    
    # (d)
    print('=== (d) ===')
    k_list =  [10, 50, 100, 200, 500]
    lamda_list = [0.001, 0.01, 0.1, 1]
    # k_opt_lamda_list = []
    # acc_list = []
    best_k = 0
    best_lamda = 0
    best_acc = 0
    
    
    for k in k_list:
        loop_best_acc_list = []
        for lamda in lamda_list:
            print('\n k = {0}, lamda = {1}\n'.format(k,lamda))
            model_l2 = AutoEncoder_l2(num_question=num_question, k=k) # add l2
            val_acc = (train_l2(model_l2, lr, lamb, train_matrix, zero_train_matrix,
                    valid_data, num_epoch, lamda))[0] # return each accuracy
            if best_acc < val_acc:
                best_k = k
                best_lamda = lamda
                best_acc = val_acc
                
    print('The most optimal k by training is {}\n'.format(best_k))
    print('The most optimal lamda by training is {}\n'.format(best_lamda))
    
    print('The final validation accuracy is {}\n'.format(best_acc))
    


    test_accs = evaluate(model_l2, zero_train_matrix, test_data) 
    print('The final test accuracy is {}'.format(test_accs))
    
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

if __name__ == "__main__":
    main()
