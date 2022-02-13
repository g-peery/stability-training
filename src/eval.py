from PIL import Image
from loader import val_ldr
from trainer import RobustInception
from torchvision import transforms
from tqdm import tqdm
import torch

if torch.cuda.is_available():
    device= torch.device('cuda') 
else:
    device= torch.device('cpu') 
print(f"using: {device}")

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

def evaluate(model_fpath):
    
    # load model 
    model=  RobustInception().to(device)
    model.load_state_dict(torch.load(model_fpath))
    model.eval()

    print("length of valid set", len(val_ldr))
    val_labels = []
    for (imgs, labels) in tqdm(val_ldr,
                                desc="iteration",
                                unit="%",
                                disable=True):
        imgs = imgs.to(device)
        labels = labels.to(device)
        
        preds = model(imgs)
        val_labels.append(labels)
        print(f"preds: {preds}, val_labels:{val_labels}")

if __name__=='__main__':
    model_fpath = "/home/abaruwa/CIS_572/model_weights_22.pth"
    evaluate(model_fpath)
