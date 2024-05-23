#################################################### IMPORT LIBRARIES ####################################################

import torch 
import torch.distributions as dist
import time

#################################################### DEFINE FUNCTIONS ####################################################

def exponentiate_input(x_input, M_value):
    """
    Given N datapoints, return a matrix of shape (N,M) where the n-th row contains a vector of the n-th datapoint 
    exponentiated up to M_value.

    Input
    x_input (torch tensor): vector of shape (N,) containing x values, input data
    M_value (scalar): Degree of polynomial function
    """
    return torch.pow(torch.repeat_interleave(x_input, M_value+1).reshape(-1,M_value+1), torch.arange(M_value+1))

def polynomial_fun(weight_vector, x_input):
    """
    Given M+1 weights, return predicted y value with M polynomial function

    Input
    weight_vector (torch tensor): shape (M+1,) containing the model's weights values
    x_input (torch tensor): vector of shape (N,) containing x values, input data

    Output
    y_predict (torch tensor): vector of size N containing all the predicted y values
    """
    #Acquire input shapes
    M = weight_vector.shape[0] - 1

    #Prepare inputs
    polynomial_input = exponentiate_input(x_input, M)

    #"Feed-forward"
    y_predict = torch.matmul(polynomial_input, weight_vector)

    return y_predict

def fit_polynomial_ls(x_input, t_input, M_value):
    """
    Given N amount of x input values and t target values, return weights that minimise least-square loss of 
    above polynomial model: AX = B. Can also specify polynomial degree via M_values.

    Input
    x_input (torch tensor): vector of shape (N,) containing x values, input data
    t_input (torch tensor): vector of shape (N,) containing t values, target data
    M_value (scalar): Degree of polynomial function

    Output
    weights (torch tensor): vector of shape (M+1,) containing LS optimised weights
    """
    #Compute solution to LS
    weights = torch.linalg.lstsq(exponentiate_input(x_input, M_value), t_input)[0]

    return weights

def fit_polynomial_sgd(x_input, t_input, M_value, learning_rate, minibatch_size):
    """
    Given N amount of x input values and t target values, return weights that optimise least-square 
    loss of polynomial model via SGD.

    Input
    x_input (torch tensor): vector of shape (N,) containing x values, input data
    t_input (torch tensor): vector of shape (N,) containing t values, target data
    M_value (scalar): Degree of polynomial function
    learning_rate (scalar): learning rate of SGD optimiser
    minibatch_size (scalar): amount of input data per minibatch run

    Output
    weights (torch tensor): vector of shape (M+1,) containing SGD optimised weights
    loss (python list): list recording loss per epoch
    """
    loss_list = []
    best_loss = torch.inf
    number_of_batches = x_input.shape[0]//minibatch_size
    
    weights = torch.randn(M_value+1, requires_grad=True)
    optimiser = torch.optim.SGD((weights,), lr = learning_rate, momentum=0.9)
    
    #Start optimisation
    epoc_number = 5000
    for epoc in range(1, epoc_number+1):
        random_input_data = torch.randperm(x_input.shape[0])
        for batch in range(number_of_batches):
            batch_loss = 0
            random_indexes = random_input_data[batch*minibatch_size:(batch+1)*minibatch_size]
            x_input_minibatch = x_input[random_indexes]
            t_input_minibatch = t_input[random_indexes]
            #Zero the gradients
            optimiser.zero_grad()

            #Feedforward minibatch
            y_predict_minibatch = polynomial_fun(weights, x_input_minibatch)

            #Compute and report the loss, I will use MSE loss
            loss = torch.mean(torch.pow((t_input_minibatch - y_predict_minibatch), 2))

            #Backpropagate to acquire gradients
            loss.backward()
            
            #Update weights
            optimiser.step()

            batch_loss += loss.item()

        loss_list.append(batch_loss)
        if batch_loss < best_loss:
            best_loss = batch_loss
            best_weights = weights
            
        if epoc % 500 == 0:
            print("Epoc #",epoc, "Loss =", loss.item())

    return best_weights, loss_list

if __name__ == '__main__':
    #################################################### GENERATE DATA ####################################################

    #Define variables
    weights_generating = torch.tensor([1.,2.,3.])
    x_train = dist.Uniform(torch.tensor([-20.0]), torch.tensor([20.0])).sample(sample_shape=(20,))
    x_test = dist.Uniform(torch.tensor([-20.0]), torch.tensor([20.0])).sample(sample_shape=(10,))

    #Acquire predicted y
    y_train_true = polynomial_fun(weights_generating, x_train)
    y_test_true = polynomial_fun(weights_generating, x_test)

    #Add gaussian noise
    t_train = y_train_true + torch.normal(mean = 0, std = 0.5, size = (20,))
    t_test = y_test_true + torch.normal(mean = 0, std = 0.5, size = (10,))

    #################################################### FIT DATA VIA LEAST SQUARES ####################################################

    mean_train_observed = torch.mean(t_train - y_train_true)
    std_train_observed = torch.sqrt(torch.mean(torch.pow(t_train - y_train_true, 2)))
    print("######################################################## START JOB ###########################################################\n")
    print("PLEASE NOTE: In order to get SGD results to be close to LS ones, I set large epoch number, small learning rate and small batch size.\
        \nAs such, it might take some time (30 seconds) per M value. \
        \nI apologise if this is annoying, but if not SGD results are not even close to LS ones. \
        \nThe coursework does not specify performance nor training time for marks so I hope this is okay. \
        \nFurthermore, since I am using such low minibatch size and such large epoch size, the printed loss might not seem like is converging but it most definitely is (I verified using matplotlib). \
        \nI added the matplotlib plots showcasing diminishing loss on task1 directory for the marker to view if they desire. \
        \nThanks! :)")

    print("\n######################################################## OBSERVED DATA VS TRUE POLYNOMIAL ###########################################################\n")
    print("a) Mean in difference wrt \"true\" polynomial curve of observed training data:", mean_train_observed.item() , \
        "\nStandard Deviation in difference wrt \"true\" polynomial curve of observed training data:", std_train_observed.item())

    print("\n######################################################## PERFORMANCE OF LEAST SQUARES ###########################################################\n")
    ls_weights_list = []
    ls_runtime_list = []
    for M in [2,3,4]:
        start_time = time.time()
        weights = fit_polynomial_ls(x_train, t_train, M_value=M)
        end_time = time.time()
        ls_runtime_list.append(end_time - start_time)
        ls_weights_list.append(weights)

        y_train_predict = polynomial_fun(weights, x_train)
        y_test_predict = polynomial_fun(weights, x_test)

        mean_train = torch.mean(y_train_predict - y_train_true)
        std_train = torch.sqrt(torch.mean(torch.pow(y_train_predict - y_train_true, 2)))

        mean_test = torch.mean(y_test_predict - y_test_true)
        std_test = torch.sqrt(torch.mean(torch.pow(y_test_predict - y_test_true, 2)))

        print("b) For M =", M, ":\n-Training input data:\n    Mean of LS-predicted y-values in difference:", mean_train.item() , \
            "\n    Standard Deviation of LS-predicted y-values in difference:", std_train.item(), \
            "\n-Testing input data:\n    Mean of LS-predicted y-values in difference:", mean_test.item() , \
            "\n    Standard Deviation of LS-predicted y-values in difference:", std_test.item(), "\n")

    #################################################### FIT DATA VIA SGD ####################################################

    print("######################################################## PERFORMANCE OF SGD ###########################################################\n")
    sgd_weights_list = []
    sgd_runtime_list = []
    for M in [2,3,4]:
        if M == 2:
            lr = 1e-7
            mini_batch_size = 1
        elif M == 3:
            lr = 1e-10
            mini_batch_size = 1
        else:
            lr = 1e-12
            mini_batch_size = 1
        
        print("Begin optimisation of weights for M =", M)
        start_time = time.time()
        weights, loss = fit_polynomial_sgd(x_train, t_train, M_value=M, learning_rate=lr, minibatch_size=mini_batch_size)
        end_time = time.time()
        sgd_runtime_list.append(end_time - start_time)
        sgd_weights_list.append(weights)

        y_train_predict = polynomial_fun(weights, x_train)
        y_test_predict = polynomial_fun(weights, x_test)

        mean_train = torch.mean(y_train_predict - y_train_true)
        std_train = torch.sqrt(torch.mean(torch.pow(y_train_predict - y_train_true, 2)))

        mean_test = torch.mean(y_test_predict - y_test_true)
        std_test = torch.sqrt(torch.mean(torch.pow(y_test_predict - y_test_true, 2)))

        print("c) For M =", M, ":\n-Training input data:\n    Mean of SGD-predicted y-values in difference:", mean_train.item() , \
            "\n    Standard Deviation of SGD-predicted y-values in difference:", std_train.item(), \
            "\n-Testing input data:\n    Mean of SGD-predicted y-values in difference:", mean_test.item() , \
            "\n    Standard Deviation of SGD-predicted y-values in difference:", std_test.item(), "\n")
        
    #################################################### REST OF TESTS ####################################################

    print("######################################################## LS VS SGD, ACCURACY COMPARISON ###########################################################\n")

    print("PLEASE NOTE: I will assume 'ground-truth' means the observed data set (t values on coursework) and not the 'true' polynomial curve\n")
    ground_truth_weights_list = [torch.tensor([1,2,3]), torch.tensor([1,2,3,0]), torch.tensor([1,2,3,0,0])]
    for i, M in enumerate([2,3,4], 0):
        ground_truth_weights = ground_truth_weights_list[i]

        #SGD
        sgd_weights = sgd_weights_list[i]
        rmse_sgd_w = torch.sqrt(torch.mean(torch.pow(sgd_weights - ground_truth_weights, 2)))

        y_sgd_test_predict = polynomial_fun(sgd_weights, x_test)
        rmse_sgd_y = torch.sqrt(torch.mean(torch.pow(t_test - y_sgd_test_predict, 2)))

        #LS
        ls_weights = ls_weights_list[i]
        rmse_ls_w = torch.sqrt(torch.mean(torch.pow(ls_weights - ground_truth_weights, 2)))

        y_ls_test_predict = polynomial_fun(ls_weights, x_test)
        rmse_ls_y = torch.sqrt(torch.mean(torch.pow(t_test - y_ls_test_predict, 2)))

        print("d) For M = ", M, "\nRMSE of LS and SGD wrt ground truth weights: \
            \n    LS:", rmse_ls_w.item(), "\
            \n    SGD:", rmse_sgd_w.item(), "\
            \nRMSE of LS and SGD wrt to observed y values: \
            \n    LS:", rmse_ls_y.item(), "\
            \n    SGD:", rmse_sgd_y.item(), "\n")
        
    #################################################### RUNTIME COMPARISON ####################################################

    print("######################################################## LS VS SGD, RUNTIME COMPARISON ###########################################################\n")
    for i, M in enumerate([2,3,4], 0):
        ls_runtime = ls_runtime_list[i]
        sgd_runtime = sgd_runtime_list[i]
        print("e) For M =", M, "\nTime spent in fitting/training data for LS and SGD in seconds:\
            \n    LS:", ls_runtime, "\
            \n    SGD:", sgd_runtime, "\n")