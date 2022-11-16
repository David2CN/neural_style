import requests
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO

import torch
from torchvision import transforms
from tqdm import tqdm


def load_image(img_path, max_size=400, shape=None):
    ''' Load in and transform an image, making sure the image
       is <= 400 pixels in the x-y dims.'''
    if "http" in img_path:
        response = requests.get(img_path)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(img_path).convert('RGB')
    
    # large images will slow down processing
    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)
    
    if shape is not None:
        size = shape
        
    in_transform = transforms.Compose([
                        transforms.Resize(size),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), 
                                             (0.229, 0.224, 0.225))])

    # discard the transparent, alpha channel (that's the :3) and add the batch dimension
    image = in_transform(image)[:3,:,:].unsqueeze(0)
    
    return image

    
# convert from a Tensor image to a NumPy image for display
def im_convert(tensor):
    """ Display a tensor as an image. """
    
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1,2,0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)

    return image


def get_features(image, model, layers=None):
    """ Run an image forward through a model and get the features for 
        a set of layers. Default layers are for VGGNet matching Gatys et al (2016)
    """
    
    if layers is None:
        layers = {'0': 'conv1_1',
                  '5': 'conv2_1', 
                  '10': 'conv3_1', 
                  '19': 'conv4_1',
                  '21': 'conv4_2',
                  '28': 'conv5_1'}        
        
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
            
    return features


def gram_matrix(tensor):
    """ Calculate the Gram Matrix of a given tensor 
        Gram Matrix: https://en.wikipedia.org/wiki/Gramian_matrix
    """
    
    ## get the batch_size, depth, height, and width of the Tensor
    ## reshape it, so we're multiplying the features for each channel
    ## calculate the gram matrix
    
    b, d, h, w = tensor.size()
    tensor = tensor.view(b*d, h*w)
    gram = torch.mm(tensor, tensor.t())
    
    return gram 


def train(model, content, style, style_weights: dict, content_loss_layer: str, steps:int=5000, content_weight: float=1, style_weight: float=1e6, show_every: int=None):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # get content and style features only once before forming the target image
    content_features = get_features(content, model)
    style_features = get_features(style, model)

    # calculate the gram matrices for each layer of our style representation
    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

    # target images
    target = content.clone().requires_grad_(True).to(device)

    # iteration hyperparameters
    optimizer = torch.optim.Adam([target], lr=0.003)

    for step in tqdm(range(1, steps+1)):
        
        # get the features from your target image
        target_features = get_features(target, model)
        
        # the content loss
        content_loss = torch.mean((target_features[content_loss_layer] - content_features[content_loss_layer])**2)
        
        # the style loss
        # initialize the style loss to 0
        style_loss = 0
        # then add to it for each layer's gram matrix loss
        for layer in style_weights:
            # get the "target" style representation for the layer
            target_feature = target_features[layer]
            target_gram = gram_matrix(target_feature)
            _, d, h, w = target_feature.shape
            # get the "style" style representation
            style_gram = style_grams[layer]
            # the style loss for one layer, weighted appropriately
            layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram)**2)
            # add to the style loss
            style_loss += layer_style_loss / (d * h * w)
            
        # calculate the *total* loss
        total_loss = content_weight * content_loss + style_weight * style_loss
        
        # update your target image
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # display intermediate images and print the loss
        if show_every:
            if  step % show_every == 0:
                print('Total loss: ', total_loss.item())
                plt.imshow(im_convert(target))
                plt.show()
        
    return target



