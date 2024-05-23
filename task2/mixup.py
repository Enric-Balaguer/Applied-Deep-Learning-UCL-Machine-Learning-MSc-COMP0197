############################################### IMPORT LIBRARIES ###############################################

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
from PIL import Image

#Seed for reproducible results
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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
    
def generate_example_images():
    """
    This function generates example augmented images for both sampling methods.
    """
    ## cifar-10 dataset
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 16
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    ## example images NON_AUGMENTED
    dataiter = iter(trainloader)
    images_real, labels_real = next(dataiter) # note: for pytorch versions (<1.14) use dataiter.next()
    im = Image.fromarray((torch.cat(images_real.split(1,0),3).squeeze()/2*255+.5*255).permute(1,2,0).numpy().astype('uint8'))
    im.save("mixup_notaugmented.png")
    print('mixup_notaugmented.png saved.')

    sampling_method_list = [1,2]
    for i, sampling_method in enumerate(sampling_method_list):
        ## data augmentation method
        Augmentation_Mixup = MixUp(sampling_method=sampling_method, alpha=0.4)

        if sampling_method == 1:
            ## example images beta sampling augmented
            images, labels = Augmentation_Mixup.generate_augmentation(image_batch=images_real, label_batch=labels_real)
            im = Image.fromarray((torch.cat(images.split(1,0),3).squeeze()/2*255+.5*255).permute(1,2,0).numpy().astype('uint8'))
            im.save("mixup_beta.png")
            print('mixup_beta.png saved.')

        if sampling_method == 2:
            ## example images uniform sampling augmented
            images, labels = Augmentation_Mixup.generate_augmentation(image_batch=images_real, label_batch=labels_real)
            im = Image.fromarray((torch.cat(images.split(1,0),3).squeeze()/2*255+.5*255).permute(1,2,0).numpy().astype('uint8'))
            im.save("mixup_uniform.png")
            print('mixup_uniform.png saved.')

############################################### GENERATE IMAGES ###############################################

if __name__ == '__main__':
        
    #Generate normal images and augmented images with both sampling methods
    generate_example_images()
