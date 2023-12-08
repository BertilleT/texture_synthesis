import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
#transforms
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from utils import *
import matplotlib.pyplot as plt

def textureGeneration(epochs, synth, cnn, loss, target, gramm_targets, outputs, layers, optimizer):
    total_loss = None 
    def closure():
        nonlocal epoch, total_loss

        optimizer.zero_grad()

        # Forward pass using synth. Get activations of selected layers for image synth (outputs).
        cnn(synth)
        gram_synth = [gram(outputs[key]) for key in layers]

        # Compute loss for each activation
        losses = []
        for synth_gram, target_gram in zip(gram_synth, gramm_targets):
            losses.append(loss(synth_gram, target_gram))
        total_loss = sum(losses)
        total_loss.backward()

        #if epoch % 10 == 0:
        return total_loss.item()

    for epoch in range(1, epochs + 1):
        tl = optimizer.step(closure)
        print("Epoch: ", epoch)
        print("Loss: ", tl)
        plot_synth_img(target, synth, epoch, tl)
    return synth