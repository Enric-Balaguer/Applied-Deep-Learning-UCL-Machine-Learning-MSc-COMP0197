#################################################### IMPORT LIBRARIES ####################################################

import torch 
import torch.distributions as dist
from task import polynomial_fun, fit_polynomial_ls

#################################################### DEFINE FUNCTIONS ####################################################

def fit_polynomial_sgd_learnable_M(x_input, t_input, learning_rate, minibatch_size):
    """
    Given N amount of input values and t target values, first determine best degree of polynomial curve to fit to data
    via SGD and then return optimised weights according to least squares.

    Input
    x_input (torch tensor): vector of shape (N,) containing x values, input data
    t_input (torch tensor): vector of shape (N,) containing t values, target data
    learning_rate (scalar): learning rate of SGD optimiser
    minibatch_size (scalar): amount of input data per minibatch run

    Output
    best_M (scalar): optimised degree of polynomial curve to fit the data with
    best_weights (torch tensor): vector with optimised weights according to least squares
    loss_list (python list): python list containing loss in every 100th epoch
    M_list (python list): python list containing optimised M_value per epoch (was used for plotting purposes)
    """
    loss_list = []
    M_value_list = []
    number_of_batches = x_input.shape[0]//minibatch_size

    M = torch.FloatTensor([x_input.shape[0]/2]) #Make max attainable M half the amount of input data
    M.requires_grad_(True)
    weights = torch.randn(x_input.shape[0]+1, requires_grad=True) #Weights as long as possible highest M value

    optimiser = torch.optim.SGD((weights,M), lr=learning_rate, momentum=0.9)

    #Start optimisation
    epoc_number = 1000
    for epoc in range(1, epoc_number+1):
        random_input_data = torch.randperm(x_input.shape[0])
        for batch in range(number_of_batches):
            random_indexes = random_input_data[batch*minibatch_size:(batch+1)*minibatch_size]
            x_input_minibatch = x_input[random_indexes]
            t_input_minibatch = t_input[random_indexes]

            weights_to_erase = (torch.nn.functional.gelu(M - torch.arange(x_input.shape[0]+1).float())).clamp(0,1)
            new_weights = weights * weights_to_erase

            #Feedforward begins
            optimiser.zero_grad()

            y_predict_minibatch = polynomial_fun(new_weights, x_input_minibatch)
            loss = torch.mean(torch.pow((t_input_minibatch - y_predict_minibatch), 2))

            loss.backward()

            torch.nn.utils.clip_grad_norm_(weights, max_norm=1)
            torch.nn.utils.clip_grad_norm_(M, max_norm=1)

            optimiser.step()

        M_value_list.append(torch.detach(M.int()).item())
        loss_list.append(torch.detach(loss))
        
        if epoc % 100 == 0:
            print("Epoc #",epoc, "Loss =", loss.item())

    best_M = max(M_value_list, key=M_value_list.count)
    best_weights = fit_polynomial_ls(x_input, t_input, best_M)

    return best_M, best_weights, loss_list, M_value_list

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

#################################################### FIT VIA SGD ####################################################

print("################################################ START JOB ################################################")
print("\nFirst optimise M value:")
print("I am using very low batchsize, large learning rate and large epoch number, the loss might not look like it's decreasing but it most definitely is:")

lr = 0.01
minibatch_sizes = 1

best_M, best_weights, loss_list, M_value_list = fit_polynomial_sgd_learnable_M(x_train, t_train, learning_rate=lr, minibatch_size=minibatch_sizes)

y_train_predict = polynomial_fun(best_weights, x_train)
y_test_predict = polynomial_fun(best_weights, x_test)

mean_train = torch.mean(y_train_predict - y_train_true)
std_train = torch.sqrt(torch.mean(torch.pow(y_train_predict - y_train_true, 2)))

mean_test = torch.mean(y_test_predict - y_test_true)
std_test = torch.sqrt(torch.mean(torch.pow(y_test_predict - y_test_true, 2)))

print("\nOptimised value of M =", best_M)
print("\nMean and standard deviation in difference between model-predicted values and 'true' polynomial curve: \
      \nTraining data: \
      \n    Mean:", mean_train.item(), "\
      \n    Standard Deviation:", std_train.item(), "\
      \nTesting data: \
      \n    Mean:", mean_test.item(), "\
      \n    Standard Deviation:", std_test.item())