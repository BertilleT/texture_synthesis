import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
#transforms
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt



# Define parameters
model_path = '../model/vgg_conv.pth'
img = '../img/tissu.png'    
img_size = 512
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
layers = [1, 6, 11, 20, 29]
layers_weights = [1/n**2 for n in [64,128,256,512,512]]

# Def download_img
def download_img(img_name, img_size):
    loader = transforms.Compose([
        transforms.Resize(img_size),  # scale imported image
        transforms.ToTensor()])  # transform it into a torch tensor
    img = Image.open(img_name)
    img = loader(img).unsqueeze(0)
    return img.to(device, torch.float)

#Take features map as input and compute the gram matrix of it
def gram(features_map):
    batch, nb_features_map, h,w = features_map.size()
    F = features_map.view(batch, nb_features_map, h*w)
    G = torch.bmm(F, F.transpose(1,2))
    G.div_(h*w)
    return G

# Def hook 
def save_features_map(layer, fm):
    def hook(module, module_in, module_out): 
        fm[layer] = module_out
    return hook

# fct from the TP
def plot_synth_img(target, synth_img, epoch, loss):
    """ Displays the intermediate results of the main iteration
    """
    print('Iteration: %d, loss: %f'%(epoch, loss.item()))
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 20))
    axes[0].imshow(target.cpu().detach().squeeze(0).permute(1, 2, 0).numpy())
    axes[0].set_title('Original texture')
    axes[0].axis('off')
    axes[1].imshow(synth_img.cpu().detach().squeeze(0).permute(1, 2, 0).numpy())
    axes[1].set_title('Synthesis')
    axes[1].axis('off')
    fig.tight_layout()
    plt.pause(0.05)