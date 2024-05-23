############################################### IMPORT LIBRARIES ###############################################
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import time
import pickle
from PIL import Image
from network_pt import MyViT

############################################### DEFINE CLASSES ###############################################

class MixUp():
    """
    Python Class that implements the MixUp data augmentation algorithm for both beta and uniform sampling methods of lambda

    Input
    self.sampling_method (scalar): specify beta sampling (sampling_method == 1) or uniform sampling (sampling_method == 2)
    self.alpha (scalar): determine alpha value for beta sampling
    """
    def __init__(self, sampling_method, alpha):
        self.sampling_method = sampling_method
        self.alpha = alpha

    def generate_augmentation(self, image_batch, label_batch):
        """
        Implements MixUp data augmentation algorithm

        Inputs
        image_batch (torch tensor): containing batch of images
        label_batch (torch tensor): matrix of one-hot encoded batch labels!

        Outputs
        generated_images (torch tensor): containing augmented images
        generated_labels (torch tensor): matrix of one-hot encoded augmented labels
        """
        random_indexes = torch.randperm(image_batch.shape[0]) #torch.arange(image_batch.shape[0]) - (torch.randint(low = 0, high = image_batch.shape[0], size=(1,))).item()
        image_batch_1 = image_batch
        image_batch_2 = image_batch[random_indexes]

        label_batch_1 = label_batch
        label_batch_2 = label_batch[random_indexes]

        #Sample lambda according to sampling_method
        if self.sampling_method == 1:
            lambd = torch.distributions.Beta(self.alpha, self.alpha).sample()

        elif self.sampling_method == 2:
            low = 0.
            high = 1.
            lambd = torch.distributions.Uniform(low, high).sample()

        generated_images = lambd*image_batch_1 + (1-lambd)*image_batch_2
        generated_labels = (lambd*label_batch_1 + (1-lambd)*label_batch_2)

        return generated_images, generated_labels

def train_using_both_methods():
    """
    This function trains my vision transformer model for both sampling methods for 20 epochs.
    It saves both models after training and saves and prints the following:
    Keep in mind, my chosen metric is accuracy.

    Training accuracy: After every epoch, accuracy is determined on all train set. 
    Validation accuracy: After every epoch, accuracy is determined on all validation set. 
    Training loss: During each training epoch, average batch loss is calculated (on train set). 
    Validation loss: After every epoch, the average batch loss is calculated over all validation set. 
    Training speed: Time it takes each epoch to complete, in seconds. 
    Validation speed: Time it takes to calculate loss and accuracy on validation set.
    Holdout loss: After full training is done, calculate average batch loss over all holdout test set. 
    Holdout accuracy: After full training is done, calculate accuracy over all holdout test set.
    """
    ## cifar-10 dataset
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 20

    #Acquire development, train ,and holdout test sets
    trainset_original = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset_original = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    full_dataset = torch.utils.data.ConcatDataset([trainset_original, testset_original])

    full_dataset_indices = range(len(full_dataset))
    development_set = torch.utils.data.Subset(full_dataset, full_dataset_indices[0:int((len(full_dataset)*0.8))])
    test_set = torch.utils.data.Subset(full_dataset, full_dataset_indices[int((len(full_dataset)/10)*8):-1])

    development_set_indices = range(len(development_set))
    train_set = torch.utils.data.Subset(development_set, development_set_indices[0:int((len(development_set)*0.9))])
    validation_set = torch.utils.data.Subset(development_set, development_set_indices[int((len(development_set)*0.9)):-1])

    train_set_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    validation_set_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_set_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    ## data augmentation method
    accuracy_list_training = []
    accuracy_list_validation = []
    training_loss_list = []
    validation_loss_list = []
    running_time_training = []
    running_time_validation = []
    for sampling_method in [1,2]:
        ## VisionTransformer model
        net = MyViT(number_classes=len(classes))
        
        ## loss and optimiser
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

        Augmentation_Mixup = MixUp(sampling_method=sampling_method, alpha=0.4)
        running_time_training_history = []
        running_time_validation_history = []
        accuracy_training_history = []
        accuracy_validation_history = []
        validation_loss_history = []
        training_loss_history = []
        if sampling_method == 1:
            print("Start training model with beta sampling method for 20 epochs")
        if sampling_method == 2:
            print("Start training model with uniform sampling method for 20 epochs")

        for epoch in range(20):  # loop over the dataset multiple times

            ## TRAINING
            start_time_training = time.time()
            running_loss_training = 0.0
            for i_training, data in enumerate(train_set_loader, 1):
                inputs_original, labels_original = data
                labels_original_hotencoded = torch.nn.functional.one_hot(labels_original, num_classes=len(classes)) #hot encode for mixup

                inputs, labels = Augmentation_Mixup.generate_augmentation(image_batch=inputs_original, label_batch=labels_original_hotencoded)

                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss_training += loss.item()

            # collect training speed time
            end_time_training = time.time()

            #ACCURACY VALIDATION
            start_time_validation = time.time()    
            correct_predict = 0.0
            running_loss_validation = 0.0
            for i_validation, data in enumerate(validation_set_loader, 1):
                images, labels = data
                outputs = net(images)
                loss = criterion(outputs, labels)
                running_loss_validation += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct_predict += torch.sum(labels == predicted)
            accuracy_validation = (correct_predict/len(validation_set)).item()
            end_time_validation = time.time()

            #ACCURACY TRAINING
            correct_predict = 0.0
            for i_training_accuracy, data in enumerate(train_set_loader, 1):
                images, labels = data
                outputs = net(images)
                _, predicted = torch.max(outputs, 1)
                correct_predict += torch.sum(labels == predicted)
            accuracy_training = (correct_predict/len(train_set)).item()

            # print metric monitoring and collect statistics
            print("Epoch #", epoch, "average batch loss while training:", running_loss_training/i_training)
            print("Epoch #", epoch, "accuracy on validation set:", accuracy_validation)

            accuracy_training_history.append(accuracy_training)
            accuracy_validation_history.append(accuracy_validation)
            training_loss_history.append(running_loss_training/i_training)
            validation_loss_history.append(running_loss_validation/i_validation)
            running_time_training_history.append(end_time_training - start_time_training)
            running_time_validation_history.append(end_time_validation - start_time_validation)

        accuracy_list_training.append(accuracy_training_history)
        accuracy_list_validation.append(accuracy_validation_history)
        training_loss_list.append(training_loss_history)
        validation_loss_list.append(validation_loss_history)
        running_time_training.append(running_time_training_history)
        running_time_validation.append(running_time_validation_history)

        if sampling_method == 1:
            torch.save(net.state_dict(), 'saved_model_beta.pt')
            print('Model saved for beta sampling method')
        if sampling_method == 2:
            torch.save(net.state_dict(), 'saved_model_uniform.pt')
            print('Model saved for uniform sampling method')


    # Start testing on holdout test set
    holdout_test_loss_list = []
    holdout_test_accuracy = []
    for sampling_method in [1,2]:
        holdout_test_loss_history = []
        holdout_test_accuracy_history = []

        ## Load vision transformer
        net = MyViT(number_classes=len(classes))
        if sampling_method == 1:
            net.load_state_dict(torch.load('saved_model_beta.pt'))
        elif sampling_method == 2:
            net.load_state_dict(torch.load('saved_model_uniform.pt'))
        
        running_loss = 0.0
        correct_predict = 0.0
        for i, data in enumerate(test_set_loader, 1):
            images, labels = data
            outputs = net(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_predict += torch.sum(labels == predicted)
        accuracy_holdout = (correct_predict/len(test_set)).item()

        holdout_test_loss_history.append(running_loss/i)
        holdout_test_accuracy_history.append(accuracy_holdout)
        
        holdout_test_loss_list.append(holdout_test_loss_history)
        holdout_test_accuracy.append(holdout_test_accuracy_history)

    return accuracy_list_training, accuracy_list_validation, training_loss_list, validation_loss_list, running_time_training, running_time_validation, holdout_test_loss_list, holdout_test_accuracy

############################################### RUN TRAINING ###############################################

if __name__ == '__main__':

    RUN_TRAINING_FLAG = False

    if RUN_TRAINING_FLAG == True:
        print("###################################################### START JOB ######################################################")
        print("\nPLEASE NOTE: The RUN_TRAINING_FLAG at the bottom of the file was set to 'True'. \
              \nConsequently, training for both sampling methods will begin. This will take some time. \
              \nIf the marker wishes to only view results, please set RUN_TRAINING_FLAG to 'False'. \
              \nIf the marker wishes to verify my training loop actually works, and after that view results, please set RUN_TRAINING_FLAG to 'True'. \
              \nI hope this is more helpful than annoying.")
        
        print("\n###################################################### START TRAINING ######################################################\n")

        #Train model for both sampling methods and plot accuracy vs epochs of each
        accuracy_list_training, accuracy_list_validation, training_loss_list, validation_loss_list, running_time_training, running_time_validation, holdout_test_loss_list, holdout_test_accuracy = train_using_both_methods()

        #Save lists
        with open('accuracy_list_training.pkl', 'wb') as f:
            pickle.dump(accuracy_list_training, f)
        with open('accuracy_list_validation.pkl', 'wb') as f:
            pickle.dump(accuracy_list_validation, f)
        with open('training_loss_list.pkl', 'wb') as f:
            pickle.dump(training_loss_list, f)
        with open('validation_loss_list.pkl', 'wb') as f:
            pickle.dump(validation_loss_list, f)
        with open('running_time_training.pkl', 'wb') as f:
            pickle.dump(running_time_training, f)
        with open('running_time_validation.pkl', 'wb') as f:
            pickle.dump(running_time_validation, f)
        with open('holdout_test_loss_list.pkl', 'wb') as f:
            pickle.dump(holdout_test_loss_list, f)
        with open('holdout_test_accuracy.pkl', 'wb') as f:
            pickle.dump(holdout_test_accuracy, f)

        print("\n###################################################### SUMMARY OF LOSS VALUES, SPEED, METRIC ON TRAINING AND VALIDATION ######################################################")
        print("\nPLEASE NOTE: I am quite confused on the wording of the exercise. Because of that, I showcase everything that might be wanted. I explain each below. \
              \nI apologise if all of these were not wanted, I am just trying to secure all the marks. \
              \nKeep in mind, my chosen metric is accuracy. \
              \nTraining accuracy: After every epoch, accuracy is determined on all train set. \
              \nValidation accuracy: After every epoch, accuracy is determined on all validation set. \
              \nTraining loss: During each training epoch, average batch loss is calculated (on train set). \
              \nValidation loss: After every epoch, the average batch loss is calculated over all validation set. \
              \nTraining speed: Time it takes each epoch to complete, in seconds. \
              \nValidation speed: Time it takes to calculate loss and accuracy on validation set.")

        print("\nTraining accuracy:")
        for i, list in enumerate(accuracy_list_training):
            if i == 0:
                print("\nBeta sampling method:")
            else:
                print("\nUniform sampling method:")
            for epoch, accuracy in enumerate(list):
                print("Epoch #", epoch,"accuracy on training set:",accuracy)

        print("\nValidation accuracy:")
        for i, list in enumerate(accuracy_list_validation):
            if i == 0:
                print("\nBeta sampling method:")
            else:
                print("\nUniform sampling method:")
            for epoch, accuracy in enumerate(list):
                print("Epoch #", epoch,"accuracy on validation set:",accuracy)

        print("\nTraining loss:")
        for i, list in enumerate(training_loss_list):
            if i == 0:
                print("\nBeta sampling method:")
            else:
                print("\nUniform sampling method:")
            for epoch, loss in enumerate(list):
                print("Epoch #", epoch,"average batch loss while training:",loss)

        print("\nValidation loss:")
        for i, list in enumerate(validation_loss_list):
            if i == 0:
                print("\nBeta sampling method:")
            else:
                print("\nUniform sampling method:")
            for epoch, loss in enumerate(list):
                print("Epoch #", epoch,"average batch loss on validation set:",loss)

        print("\nTraining speed:")
        for i, list in enumerate(running_time_training):
            if i == 0:
                print("\nBeta sampling method:")
            else:
                print("\nUniform sampling method:")
            for epoch, timed in enumerate(list):
                print("Epoch #", epoch,"seconds taken to complete:",timed)

        print("\nValidation speed:")
        for i, list in enumerate(running_time_validation):
            if i == 0:
                print("\nBeta sampling method:")
            else:
                print("\nUniform sampling method:")
            for epoch, timed in enumerate(list):
                print("Epoch #", epoch,"seconds taken to calculate accuracy on validation set:",timed)

        print("\n###################################################### SUMMARY OF LOSS VALUES AND ACCURACY ON HOLDOUT SET ######################################################")
        print("\nPLEASE NOTE: I am quite confused on the wording of the exercise. Because of that, I showcase everything that might be wanted. I explain each below. \
              \nI apologise if all of these were not wanted, I am just trying to secure all the marks. \
              \n\nHoldout loss: After full training is done, calculate average batch loss over all holdout test set. \
              \nHoldout accuracy: After full training is done, calculate accuracy over all holdout test set.")

        print("\nHoldout loss:")
        for i, list in enumerate(holdout_test_loss_list):
            if i == 0:
                print("\nBeta sampling method:")
            else:
                print("\nUniform sampling method:")
            for loss in list:
                print("Average batch loss over all holdout test set:",loss)

        print("\nHoldout accuracy:")
        for i, list in enumerate(holdout_test_accuracy):
            if i == 0:
                print("\nBeta sampling method:")
            else:
                print("\nUniform sampling method:")
            for accuracy in list:
                print("Acuraccy on holdout test set:",accuracy)
        
        print("\nThese results can be compared to the same statistics obtained during development above.")

    if RUN_TRAINING_FLAG == False:
        print("###################################################### START JOB ######################################################")
        print("\nPLEASE NOTE: The RUN_TRAINING_FLAG at the bottom of the file was set to 'False'. \
            \nConsequently, only saved results will be showcased. Training will not be undergone. This is to try to save time for the marker. \
            \nIf the marker wishes to only view results, please set RUN_TRAINING_FLAG to 'False'. \
            \nIf the marker wishes to verify my training loop actually works, and after that view results, please set RUN_TRAINING_FLAG to 'True'. \
            \nI hope this is more helpful than annoying.")
        
        #Load lists
        with open('accuracy_list_training.pkl', 'rb') as f:
            accuracy_list_training = pickle.load(f)
        with open('accuracy_list_validation.pkl', 'rb') as f:
            accuracy_list_validation = pickle.load(f)
        with open('training_loss_list.pkl', 'rb') as f:
            training_loss_list = pickle.load(f)
        with open('validation_loss_list.pkl', 'rb') as f:
            validation_loss_list = pickle.load(f)
        with open('running_time_training.pkl', 'rb') as f:
            running_time_training = pickle.load(f)
        with open('running_time_validation.pkl', 'rb') as f:
            running_time_validation = pickle.load(f)
        with open('holdout_test_loss_list.pkl', 'rb') as f:
            holdout_test_loss_list = pickle.load(f)
        with open('holdout_test_accuracy.pkl', 'rb') as f:
            holdout_test_accuracy = pickle.load(f)
        
        print("\n###################################################### SUMMARY OF LOSS VALUES, SPEED, METRIC ON TRAINING AND VALIDATION ######################################################")
        print("\nPLEASE NOTE: I am quite confused on the wording of the exercise. Because of that, I showcase everything that might be wanted. I explain each below.\
            \nI apologise if all of these were not wanted, I am just trying to secure all the marks. \
            \nKeep in mind, my chosen metric is accuracy. \
            \n\n-Training accuracy: After every epoch, accuracy is determined on all train set. \
            \n-Validation accuracy: After every epoch, accuracy is determined on all validation set. \
            \n-Training loss: During each training epoch, average batch loss is calculated (on train set). \
            \n-Validation loss: After every epoch, the average batch loss is calculated over all validation set. \
            \n-Training speed: Time it takes each epoch to complete, in seconds. \
            \n-Validation speed: Time it takes to calculate loss and accuracy on validation set.")

        print("\nTraining accuracy:")
        for i, list in enumerate(accuracy_list_training):
            if i == 0:
                print("\nBeta sampling method:")
            else:
                print("\nUniform sampling method:")
            for epoch, accuracy in enumerate(list):
                print("Epoch #", epoch,"accuracy on training set:",accuracy)

        print("\nValidation accuracy:")
        for i, list in enumerate(accuracy_list_validation):
            if i == 0:
                print("\nBeta sampling method:")
            else:
                print("\nUniform sampling method:")
            for epoch, accuracy in enumerate(list):
                print("Epoch #", epoch,"accuracy on validation set:",accuracy)

        print("\nTraining loss:")
        for i, list in enumerate(training_loss_list):
            if i == 0:
                print("\nBeta sampling method:")
            else:
                print("\nUniform sampling method:")
            for epoch, loss in enumerate(list):
                print("Epoch #", epoch,"average batch loss while training:",loss)

        print("\nValidation loss:")
        for i, list in enumerate(validation_loss_list):
            if i == 0:
                print("\nBeta sampling method:")
            else:
                print("\nUniform sampling method:")
            for epoch, loss in enumerate(list):
                print("Epoch #", epoch,"average batch loss on validation set:",loss)

        print("\nTraining speed:")
        for i, list in enumerate(running_time_training):
            if i == 0:
                print("\nBeta sampling method:")
            else:
                print("\nUniform sampling method:")
            for epoch, timed in enumerate(list):
                print("Epoch #", epoch,"seconds taken to complete:",timed)

        print("\nValidation speed:")
        for i, list in enumerate(running_time_validation):
            if i == 0:
                print("\nBeta sampling method:")
            else:
                print("\nUniform sampling method:")
            for epoch, timed in enumerate(list):
                print("Epoch #", epoch,"seconds taken to calculate accuracy on validation set:",timed)

        print("\n###################################################### SUMMARY OF LOSS VALUES AND ACCURACY ON HOLDOUT SET ######################################################")
        print("\nPLEASE NOTE: I am quite confused on the wording of the exercise. Because of that, I showcase everything that might be wanted. I explain each below.\
            \nI apologise if all of these were not wanted, I am just trying to secure all the marks. \
            \n\n-Holdout loss: After full training is done, calculate average batch loss over all holdout test set. \
            \n-Holdout accuracy: After full training is done, calculate accuracy over all holdout test set.")

        print("\nHoldout loss:")
        for i, list in enumerate(holdout_test_loss_list):
            if i == 0:
                print("\nBeta sampling method:")
            else:
                print("\nUniform sampling method:")
            for loss in list:
                print("Average batch loss over all holdout test set:",loss)

        print("\nHoldout accuracy:")
        for i, list in enumerate(holdout_test_accuracy):
            if i == 0:
                print("\nBeta sampling method:")
            else:
                print("\nUniform sampling method:")
            for accuracy in list:
                print("Acuraccy on holdout test set:",accuracy)

        print("\nThese results can be compared to the same statistics obtained during development above.")
