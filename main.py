import torch
from torchvision import models
import matplotlib.pyplot as plt
from utils import load_image, im_convert, train


# get the "features" portion of VGG19 (we will not need the "classifier" portion)
vgg = models.vgg19(weights="VGG19_Weights.DEFAULT").features

# freeze all VGG parameters since we're only optimizing the target image
for param in vgg.parameters():
    param.requires_grad_(False)

# move the model to GPU, if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
vgg.to(device)


content_dir = './images/image.jpg'
style_dir = './images/delaunay.jpg'

# load in content and style image
content = load_image(content_dir).to(device)

# Resize style to match content, makes code easier
style = load_image(style_dir, shape=content.shape[-2:]).to(device)

# weights for each style layer
style_weights = {'conv1_1': 1.,
                'conv2_1': 0.75,
                'conv3_1': 0.2,
                'conv4_1': 0.2,
                'conv5_1': 0.2}

alpha = 1  # alpha
beta = 1e6  # beta
content_loss_layer = 'conv4_2'


# transfer style
output_image = train(model=vgg, content=content, style=style, steps=10000, 
                     style_weights=style_weights, content_loss_layer=content_loss_layer,
                     content_weight=alpha, style_weight=beta)


plt.imshow(im_convert(output_image))
plt.axis("off")
plt.tight_layout()
plt.savefig("images/davidcn_delaunay.jpg", bbox_inches="tight")

