############################################### IMPORT LIBRARIES ###############################################
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
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
    It saves both models after training, prints average batch loss per epoch and prints 
    accuracy on test set per epoch
    """
    ## cifar-10 dataset
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 20

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    ## data augmentation method
    accuracy_list = []
    for sampling_method in [1,2]:

        ## VisionTransformer model
        net = MyViT(number_classes=len(classes))
        
        ## loss and optimiser
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

        Augmentation_Mixup = MixUp(sampling_method=sampling_method, alpha=0.4)
        accuracy_history = []
        if sampling_method == 1:
            print("Start training model with beta sampling method for 20 epochs")
        if sampling_method == 2:
            print("Start training model with uniform sampling method for 20 epochs")

        for epoch in range(1, 21):  # loop over the dataset multiple times
            
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs_original, labels_original = data
                labels_original_hotencoded = torch.nn.functional.one_hot(labels_original, num_classes=len(classes)) #hot encode for mixup

                # compute augmentated images and labels
                inputs, labels = Augmentation_Mixup.generate_augmentation(image_batch=inputs_original, label_batch=labels_original_hotencoded)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
            
            print("Epoch #", epoch, "Average batch loss while training:", running_loss/i)

            #Calculate accuracy
            correct_predict = 0.0
            for i, data in enumerate(testloader, 1):
                images, labels = data
                outputs = net(images)
                _, predicted = torch.max(outputs, 1)
                correct_predict += torch.sum(labels == predicted)
            accuracy = (correct_predict/len(testset)).item()
            accuracy_history.append(accuracy)
            print("Epoch #", epoch, "Accuracy on test set:", accuracy)
        
        accuracy_list.append(accuracy_history)
        if sampling_method == 1:
            torch.save(net.state_dict(), "saved_model_beta.pt")
            print("Model saved for beta sampling")
        if sampling_method == 2:
            torch.save(net.state_dict(), "saved_model_uniform.pt")
            print("Model saved for uniform sampling")

        #Implement result.png task
        results_test_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)
        for image, data in enumerate(results_test_loader):
            images, labels = data
            outputs = net(images)
            


    return accuracy_list

def generate_result_png():
    """
    Function to generate the result.png file requested in the coursework
    Coursework does not specify sampling method so I will use Beta one
    """
    ## cifar-10 dataset
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 32

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    results_test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    net = MyViT(number_classes=len(classes))
    net.load_state_dict(torch.load('saved_model_beta.pt'))

    for i, data in enumerate(results_test_loader):
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        for j in range(len(labels)):
            print("Predicted label:", predicted[j].item(), "Actual label:", labels[j].item())
        im = Image.fromarray((torch.cat(images.split(1,0),3).squeeze()/2*255+.5*255).permute(1,2,0).numpy().astype('uint8'))
        im.save("result.png")
        print("result.png saved")
        break



############################################### RUN TRAINING ###############################################

if __name__ == '__main__':

    RUN_PRETRAINING_FLAG = False

    if RUN_PRETRAINING_FLAG == True: #Train model for both sampling methods and save the models and a lists of the accuracy per epoch. Then print the results

        print("################################################### START JOB ###################################################")
        print("\nPLEASE NOTE: The RUN_PRETRAINING_FLAG variable at the end of the code is set to 'True'. \
              \nConsequently, I will proceed to train the model for both sampling methods, save both models, and report the accuracy requested in the coursework. \
              \nAlthough I have a really small model, this still takes time. \
              \nIf you want to just visualise the results, please turn the RUN_PRETRAINING_RUN to be 'False'. \
              \nI hope this is more useful than annoying")
        print("\n################################################### BEGIN TRAINING ###################################################\n")
        
        accuracy_list = train_using_both_methods() #Models are already saved while training

        with open('accuracy_list.pkl', 'wb') as f: #Save accuracy list
            pickle.dump(accuracy_list, f)

        for i, list in enumerate(accuracy_list):
            if i == 0:
                print("\nAccuracy per epoch for beta sampling method:\n")
            else:
                print("\nAccuracy per epoch for uniform sampling method:\n")
            for epoch, accuracy in enumerate(list):
                print("Epoch #", epoch,"accuracy on CIFAR-10 testset:", accuracy)

        print("\n################################################### CREATE RESULT.PNG ###################################################\n")
        generate_result_png()

    if RUN_PRETRAINING_FLAG == False: #Print results

        print("################################################### START JOB ###################################################")
        print("\nPLEASE NOTE: The RUN_PRETRAINING_FLAG variable at the end of the code is set to 'False'. \
              \nConsequently, I will only report the test set performance in terms of classification accuracy versus the epochs as requested on the coursework. \
              \nIf the marker wishes to see how training is performed, please set RUN_PRETRAINING_FLAG to be 'True'. You can find the variable at the end of the code. \
              \nI only do this because training both methods takes some time. I hope this is more useful than annoying.")
        print("\n################################################### RESULTS ###################################################\n")

        with open('accuracy_list.pkl', 'rb') as f:
            accuracy_list = pickle.load(f)
            
        for i, list in enumerate(accuracy_list):
            if i == 0:
                print("\nAccuracy per epoch for beta sampling method:\n")
            else:
                print("\nAccuracy per epoch for uniform sampling method:\n")
            for epoch, accuracy in enumerate(list):
                print("Epoch #", epoch, "accuracy on CIFAR-10  testset:", accuracy)
        
        print("\n################################################### CREATE RESULT.PNG ###################################################\n")
        generate_result_png()