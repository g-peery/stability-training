from glob import glob
from PIL import Image
from loader import val_ldr
from sklearn.metrics import precision_score
from torchmetrics import Precision
from trainer2 import Inception
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import os
import sys
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
    model=  Inception().to(device)
    model.load_state_dict(torch.load(model_fpath, map_location=device))
    model.eval()

    print("length of valid set", len(val_ldr))
    val_labels = []
    preds = []
    precision_lst = []
    precision = Precision(average='macro',num_classes=200)
   
    for (imgs, labels) in tqdm(val_ldr,
                                desc="iteration",
                                unit="%",
                                disable=True):
        imgs = imgs.to(device)
        labels = labels.to(device)

        pred_probs = model(imgs)
        
        pred_classes = torch.split(pred_probs, 200, dim=1)[0].topk(1, dim=1)[1].t().flatten()
        preds.extend(pred_classes.tolist())#.detach())#.cpu())
        val_labels.extend(labels.tolist())
        #print(f"preds: {pred_classes}, val_labels:{labels}")
        #prec_score = precision(labels, pred_classes)
        

        #precision_lst.append(prec_score)
        #print(f"preds: {pred_classes}, val_labels:{labels}, prec_score: {prec_score}")
    print(len(preds), len(val_labels)) #, val_labels.device(), preds.device())
    print("preds", preds[:100])
    print("val_labels", val_labels[:100])
    print(sum(np.array(preds) == np.array(val_labels))/len(preds))
    #prec = sum(precision_lst)/len(precision_lst)
    ##print(prec,precision_lst)
    

# Adapted from https://github.com/DennisHanyuanXu/Tiny-ImageNet/blob/master/src/data_prep.py
def create_val_img_folder(args):
    '''
    This method is responsible for separating validation images into separate sub folders
    '''
    dataset_dir = args["data_dir"]  # os.path.join(args["data_dir"], args["dataset"])
    val_dir = os.path.join(dataset_dir, 'val')
    img_dir = os.path.join(val_dir, 'images')

    fp = open(os.path.join(val_dir, 'val_annotations.txt'), 'r')
    data = fp.readlines()
    val_img_dict = {}
    for line in data:
        words = line.split('\t')
        val_img_dict[words[0]] = words[1]
    fp.close()

    # Create folder if not present and move images into proper folders
    for img, folder in tqdm(val_img_dict.items()):
        newpath = (os.path.join(img_dir, folder))
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        if os.path.exists(os.path.join(img_dir, img)):
            os.rename(os.path.join(img_dir, img), os.path.join(newpath, img))

if __name__=='__main__':
    model_fpath = sys.argv[1]
   
    data_dir = sys.argv[2]
    annotation_file = sys.argv[3]
    evaluate(model_fpath)
