from PIL import Image
from torch.optim import Adam
from tqdm import tqdm
from loader import dataloaders, val_ldr
import numpy as np
import torch
import torchvision
import torch.nn as nn
import os
import sys

if torch.cuda.is_available():
    device= torch.device('cuda') 
else:
    device= torch.device('cpu') 
print(f"using: {device}")

class Inception(nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate four parameters and assign them as
        member parameters.
        """
        super().__init__()
        # Modify the last fully connected layer
        bare_inception = torchvision.models.inception_v3(pretrained=True, aux_logits=False)
        
        for param in bare_inception.parameters():
            param.requires_grad = False
        bare_inception.fc = nn.Linear(2048, 200)
        # Incoporate in model, probability softmax incorportated into loss
        self.inception = nn.Sequential(
            bare_inception,
        )
        self.inception.add_module("softmax", nn.Softmax(dim=1))
        #print(bare_inception.named_modules)
    
    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        return self.inception(x)


def loss_fn(outputs, label): # todo: add stability loss to the training objective
    training_objective = nn.CrossEntropyLoss()
    return training_objective(outputs, label) # + ALPHA * stability_loss(distort_output, clean_output) 
    
def train(output_dir, nepochs=4):
    # Let's set up some parameters
    learning_rate=1e-4

    model = Inception().to(device)
    # We need an optimizer that tells us what form of gradient descent to do
    optimizer = Adam(model.parameters(), lr=learning_rate)

    # We also need a loss function
    LossFunction = loss_fn

    train_loader = dataloaders["train"]

    # This is default on but let's just be pedantic
    model.train()
    loss_history = []
    loss = torch.Tensor([0])
    cnt = 0
    for epoch in tqdm(range(nepochs),
                    desc=f"Epoch",
                    unit="epoch",
                    disable=False):
        for (imgs, labels) in tqdm(train_loader,
                                desc="iteration",
                                unit="%",
                                disable=True):
            optimizer.zero_grad(set_to_none=True) # Here we clear the gradients
            
            # We need to make sure the tensors are on the same device as our model
            imgs = imgs.to(device)
            labels = labels.to(device)
            out = model(imgs)
            
            loss = LossFunction(out, labels)
            
            loss.backward() 
            optimizer.step() 
            loss_history.append(loss.item())
      
        cnt += 1
        if (cnt +1) % 10 == 0:
            model_fname = os.path.join(output_dir, 'model_weights_'+str(cnt)+'.pth')
            torch.save(model.state_dict(), model_fname)
        print(f"\nEpoch {epoch}: loss: {loss.item()}")

    return model

if __name__ == '__main__':
    output_dir = sys.argv[1]
    model = train(output_dir, nepochs=100) 
    torch.save(model.state_dict(), os.path.join(output_dir, 'weights.pth'))
