import torch
import torchvision
import torch.nn as nn
from torchvision import transforms, models
import numpy as np
import argparse
from PIL import Image

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def load_image(img_path, transform=None, max_size=None, shape=None):
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
        super().__init__()
        self.select = ['0', '5', '10', '19', '28']
        self.vgg = models.vgg19(pretrained=True).features

    def forward(self, x):
        """ Extract multiple convolutional feature maps """
        features = []
        for name, layer in self.vgg._modules.items():
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
    # Make the style image same size as the content image
    content = load_image(config.content, transform, max_size=config.max_size)
    style = load_image(config.style, transform, shape=[content.size(2), content.size(3)])
    assert style.size() == content.size(), f"Size[style image:{style.size()} != content image:{content.size()}] style and content images must be of same size"

    # Initialize a target image with the content image
    target = content.clone().requires_grad_(True)

    optimizer = torch.optim.Adam([target], lr=config.lr, betas=[0.5, 0.999])
    loss = nn.MSELoss()
    vgg = VGGNet().to(device).eval()

    for step in range(config.total_step):
        # extract multiple conv feature vectors
        target_feat = vgg(target)
        content_feat = vgg(content)
        style_feat = vgg(style)

        content_loss = 0
        style_loss = 0
        for t_f, c_f, s_f in zip(target_feat, content_feat, style_feat):
            # Compute content loss with target and content images
            content_loss += loss(t_f, c_f)

            # Reshape convolutional feature maps
            _, c, h, w = t_f.size()
            t_f = t_f.view(c, h*w)
            s_f = s_f.view(c, h*w)
            
            # Compute gram matrix
            t_f = torch.mm(t_f, t_f.t())
            s_f = torch.mm(s_f, s_f.t())
            
            # Compute style loss with target and style images
            style_loss += loss(t_f, s_f) / (c * h * w)

        # Compute total loss, backprop and optimize
        total_loss = content_loss + config.style_weight * style_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if (step+1) % config.log_step == 0:
            print(f'step [{step+1}/{config.total_step}], Content Loss: {content_loss.item():.4f}, Style Loss: {style_loss.item()}, Total Loss: {total_loss.item():.4f}')

        if (step+1) % config.sample_step == 0:
            # Save the generated image
            denorm = transforms.Normalize((-2.12, -2.04, -1.80), (4.37, 4.46, 4.44))
            img = target.clone().squeeze()
            img = denorm(img).clamp_(0, 1)
            torchvision.utils.save_image(img, f'output-{step+1}.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--content', type=str, default='img/content.jpg')
    parser.add_argument('--style', type=str, default='img/style.jpg')
    parser.add_argument('--max_size', type=int, default=400)
    parser.add_argument('--total_step', type=int, default=2000)
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=500)
    parser.add_argument('--style_weight', type=str, default=100)
    parser.add_argument('--lr', type=str, default=1e-3)
    config = parser.parse_args()
    print(config)
    main(config)