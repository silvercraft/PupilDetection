from transformers import SegformerFeatureExtractor
import torch
from torch import nn
from sklearn.metrics import accuracy_score
from tqdm.notebook import tqdm
import torch
import wandb
from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
from torchvision.transforms import transforms
from transformers import SegformerForSemanticSegmentation
import json
from huggingface_hub import cached_download, hf_hub_url
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import albumentations as A

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)  
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class testDataset(Dataset):
    """Image (semantic) segmentation dataset."""

    def __init__(self, root_dir, feature_extractor):
        
        self.root_dir = root_dir
        self.feature_extractor = feature_extractor
        
        # read images
        image_file_names = []
        # for root, dirs, files in os.walk(self.root_dir):
        #   image_file_names.extend(files)
        for dir in os.listdir(self.root_dir):
            for name in os.listdir(os.path.join(self.root_dir, dir)):
                if 'jpg' in name:
                  image_file_names.append(os.path.join(self.root_dir, dir, name))
        self.images = sorted(image_file_names)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        transform = A.Compose([
            A.Resize(512, 512)
        ])
        image = Image.open(self.images[idx]).convert('RGB')

        transformed = transform(image=np.array(image))
        imaget = transformed['image']
        encoded_inputs = self.feature_extractor(imaget, return_tensors="pt")
        

        for k,v in encoded_inputs.items():
          encoded_inputs[k].squeeze_() # remove batch dimension
        
        return encoded_inputs, np.array(image), self.images[idx]

def ade_palette():
    return [[255, 255, 255], [0, 0, 0]]


same_seeds(30)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

id2label = {1: 'background', 0: 'pupil'}
label2id = {'background': 1, 'pupil': 0}

# define model
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b4",
                                                         num_labels=2, 
                                                         id2label=id2label, 
                                                         label2id=label2id,
)
print(id2label)
print(label2id)

root_dir = './data/public/S1/'
feature_extractor = SegformerFeatureExtractor(reduce_labels=True)

test_dataset = testDataset(root_dir=root_dir, feature_extractor=feature_extractor)
print("Number of testing examples:", len(test_dataset))
test_dataloader = DataLoader(test_dataset, batch_size=1, num_workers=2)

checkpoint = torch.load('./segmodel/segformer0609_b4_best.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

# os.makedirs('./segoutput/0603-1_b3', exist_ok = True)

for idx, (batch, image, img_fn) in enumerate(test_dataloader):
  # get the inputs;
  pixel_values = batch["pixel_values"].to(device)
#   print(image.shape)

  # evaluate
  with torch.no_grad():
    # forward + backward + optimize
    outputs = model(pixel_values=pixel_values)
    logits = outputs.logits.cpu()
    # print(logits.shape)

    upsampled_logits = nn.functional.interpolate(logits, size=(image.shape[1], image.shape[2]), mode="bilinear", align_corners=False)
    predicted = upsampled_logits.argmax(dim=1)

    # print(upsampled_logits.shape, predicted.shape)
    
    for i in range(predicted.shape[0]):
      color_seg = np.zeros((image.shape[1], image.shape[2], 3), dtype=np.uint8) # height, width, 3
      palette = np.array(ade_palette())
      for label, color in enumerate(palette):
          color_seg[predicted[i] == label, :] = color

      color_seg = color_seg[..., ::-1]
      color_seg = color_seg.astype(np.uint8)
      print(img_fn[i])
      # plt.figure(figsize=(15, 10))
      # plt.imshow(color_seg)
      # plt.show()
      dir_name = '/'.join(img_fn[i].split('/')[3:-1])
    #   print(dir_name)
      os.makedirs(os.path.join('./segoutput/0609_s1b4', dir_name), exist_ok = True)
      plt.imsave(os.path.join('./segoutput/0609_s1b4', '/'.join(img_fn[i].split('/')[3:])[:-3]+'png'), color_seg)

