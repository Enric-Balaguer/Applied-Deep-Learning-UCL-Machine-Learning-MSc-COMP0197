import torch.nn as nn
import torchvision.models as models

#Specify Vision Transformer model

class MyViT(nn.Module):
    def __init__(self,number_classes):
        super(MyViT, self).__init__()
        self.model = models.VisionTransformer(
            image_size=32,
            patch_size=32,
            num_layers=2,
            num_heads=2,
            hidden_dim=int(32),
            mlp_dim=int(12),
            num_classes=number_classes,
        )
    
    def forward(self,x):
        return self.model(x)