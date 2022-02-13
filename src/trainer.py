from PIL import Image
from torch.optim import SGD, Adam
from tqdm import tqdm
from loader import dataloaders
import numpy as np
import torch
import torchvision
import torch.nn as nn

if torch.cuda.is_available():
    device= torch.device('cuda') 
else:
    device= torch.device('cpu') 
print(f"using: {device}")

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.empty(tensor.size(), device=device).normal_() * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
    
class jpegCompressor():
    def __init__(self):
        """Copmpress jpeg image down to quality level, "qlty_level" """
        self.image_fpath = None
        self.qlty_level = None
        self.outfile = None

    def compress(self, image_fpath, outfile, qlty_level=100):
        img = Image.open(image_fpath)
        img.save(outfile, 
                 "JPEG", 
                 optimize = True, 
                 quality = qlty_level)
        return 

class RobustInception(nn.Module):
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
            nn.Sigmoid()
        )
        self.perturb = AddGaussianNoise(0, 0.04)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        return torch.cat((self.inception(x), self.inception(self.perturb(x))), 1)


def full_loss(outputs, label):
    ALPHA = 0.01
    training_objective = nn.CrossEntropyLoss()
    stability_loss = nn.CrossEntropyLoss()
    clean_output, distort_output = torch.split(outputs, 200, 1)
    return training_objective(clean_output, label) + ALPHA * stability_loss(distort_output, clean_output)
    
def train(nepochs=4):
    # Let's set up some parameters
    learning_rate=1e-4

    model = RobustInception().to(device)
    # We need an optimizer that tells us what form of gradient descent to do
    optimizer = Adam(model.parameters(), lr=learning_rate)

    # We also need a loss function
    LossFunction = full_loss

    train_loader = dataloaders["train"]

  
    # This is default on but let's just be pedantic
    model.train()
    loss_history = []
    loss = torch.Tensor([0])
    cnt = 0
    chk = 0
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
            
            # PyTorch is Magic!
            loss.backward() # This function calculates all our gradients
            optimizer.step() # This function does our gradient descent with those gradients
            loss_history.append(loss.item())
      
        cnt += 1
        if (cnt ) % 2 == 0:
            model_fname = '/home/abaruwa/CIS_572/model_weights_'+str(cnt)+'.pth'
            torch.save(model.state_dict(), model_fname)
        print(f"\nEpoch {epoch}: loss: {loss.item()}")

    return model

if __name__ == '__main__':
    #compressor = jpegCompressor()
    #img_qlty_levels = [80,60,40,20]
    #img_file = "/home/abaruwa/CIS_572/tiny-imagenet-200/train/n02666196/images/n02666196_294.JPEG"
    #outfile = "/home/abaruwa/CIS_572/tiny-imagenet-200/n02666196_294_10p.JPEG"
    #compressor.compress(img_file, outfile, qlty_level=10)
    model = train(nepochs=50)
    torch.save(model.state_dict(), '/home/abaruwa/CIS_572/model_weights.pth')