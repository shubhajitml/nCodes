import torch
import torchvision 
import torch.nn as nn
from torchvision import transforms, models
import numpy as np
import argparse
from PIL  import Image

# Device setup 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_image(img_path, transform = None, max_size=None, shape=None):
    """ Load an Image and convert into a torch tensor """
    img = Image.open(img_path)

    if max_size:
        scale = max_size / max(img.size)
        size = np.array(img.size) * scale
        img.resize(size.astype(int), Image.ANTIALIAS)

    if shape: 
        img.resize(shape, Image.LANCZOS)
    
    if transform:
        img = transform(img).unsqueeze(0)
    
    return img.to(device)

class VGGNet(nn.Module):
    """Select conv1_1 ~ conv5_1 activation maps."""
    def __init__(self):
        super.__init__()
        self.select = ['0', '5', '10', '19', '28']
        self.vgg = models.vgg19(pretrained=True).features

    def forward(self, x):
        """ Extract multiple convolutional feature maps """
        features = []
        for name, layer in self.vgg_modules.items():
            x = layer(x)
            if name in self.select:
                features.append(x)
        return features

def main(config):

    # VGGNet was trained on ImageNet where images are normalized by mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225].
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), 
                             std=(0.229, 0.224, 0.225))
        ]
    )

    # Load style and content image
    content = load_image(config.content, transform, max_size=config.max_size)
    style = load_image(config.content, transform, max_size=[content.size(2), content.size(3)])

    # Initialize a target image with the content image
    target = content.clone().requires_grad(True)
    
    optimizer = torch.optim.Adam([content], config.lr, betas=[0.5, 0.999])

    vgg = VGGNet().to(device).eval()

    for step in range(config.total_step):

        # extract multiple features 
        traget_features = vgg(target)
        contet_features = vgg(content)
        style_features  = vgg(style)

        

