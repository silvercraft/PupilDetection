from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
from torchvision.transforms import transforms
import albumentations as A
import torch
from albumentations.pytorch import ToTensorV2
from transformers import SegformerFeatureExtractor
from transformers import SegformerForSemanticSegmentation
import json
from torch.utils.data import DataLoader
from huggingface_hub import cached_download, hf_hub_url
from datasets import load_metric
import torch
from torch import nn
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import torch
import sys

bs = 4
lr = 0.00006

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)  
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
same_seeds(30)

class SemanticSegmentationDataset(Dataset):
    """Image (semantic) segmentation dataset."""

    def __init__(self, root_dir, feature_extractor, train=True):
        """
        Args:
            root_dir (string): Root directory of the dataset containing the images + annotations.
            feature_extractor (SegFormerFeatureExtractor): feature extractor to prepare images + segmentation maps.
            train (bool): Whether to load "training" or "validation" images + annotations.
        """
        self.root_dir = root_dir
        self.feature_extractor = feature_extractor
        self.train = train

        sub_path = "train" if self.train else "val"
        self.img_dir = os.path.join(self.root_dir, "img", sub_path)
        self.ann_dir = os.path.join(self.root_dir, "white_mask", sub_path)
        
        # read images
        image_file_names = []
        for root, dirs, files in os.walk(self.img_dir):
          image_file_names.extend(files)
        self.images = sorted(image_file_names)
        
        # read annotations
        annotation_file_names = []
        for root, dirs, files in os.walk(self.ann_dir):
          annotation_file_names.extend(files)
        self.annotations = sorted(annotation_file_names)

        assert len(self.images) == len(self.annotations), "There must be as many images as there are segmentation maps"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        transform = A.Compose([
            # A.RandomCrop(432, 576),
            A.Resize(512, 512),
            A.HorizontalFlip(p=0.5),
            # A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(),
            A.ColorJitter() 
        ])

        image = Image.open(os.path.join(self.img_dir, self.images[idx])).convert('RGB')
        segmentation_map = Image.open(os.path.join(self.ann_dir, self.annotations[idx]))

        if self.train:
            # print(np.array(image).shape, np.array(segmentation_map).shape)
            transformed = transform(image=np.array(image), mask=np.array(segmentation_map)//255)
            image = transformed['image']
            segmentation_map = transformed['mask']
        else:
            image = np.array(image)
            segmentation_map = np.array(segmentation_map)//255

        # randomly crop + pad both image and segmentation map to same size
        encoded_inputs = self.feature_extractor(image, segmentation_map, return_tensors="pt")

        for k,v in encoded_inputs.items():
          encoded_inputs[k].squeeze_() # remove batch dimension
        encoded_inputs["labels"] = (encoded_inputs["labels"]/255).to(torch.long)

        return encoded_inputs


root_dir = sys.argv[1]
feature_extractor = SegformerFeatureExtractor(reduce_labels=True)

train_dataset = SemanticSegmentationDataset(root_dir=root_dir, feature_extractor=feature_extractor)
valid_dataset = SemanticSegmentationDataset(root_dir=root_dir, feature_extractor=feature_extractor, train=False)

train_dataloader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=2)
valid_dataloader = DataLoader(valid_dataset, batch_size=bs, num_workers=2)


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

resume = False

# exit()

# wandb.init(project="CVfinal", name='segformer_b4_0604-2')
# wandb.config["batch_size"] = bs
# wandb.config["lr"] = lr

# define optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
# move model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
best_iou = 0
epoch_start = 0

model.to(device)
# if resume:
#   checkpoint = torch.load('./segmodel/segformer0604-2_b4_last.pth', map_location=device)
#   model.load_state_dict(checkpoint['model_state_dict'])
#   optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#   epoch_start = checkpoint['epoch']+1
#   best_iou = checkpoint['best_iou']


# +
for epoch in range(epoch_start, 10):  # loop over the dataset multiple times
   print("\nEpoch:", epoch)
   model.train()
  #  metric = load_metric("mean_iou", experiment_id=0)
   for idx, batch in enumerate(tqdm(train_dataloader)):
        # get the inputs;
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(pixel_values=pixel_values, labels=labels)
        loss, logits = outputs.loss, outputs.logits
        
        loss.backward()
        optimizer.step()

        if (idx+1)%1000==0:
          
          metric = load_metric("mean_iou")
          #  model.eval()
          for idx, batch in enumerate(tqdm(valid_dataloader)):
                # get the inputs;
                pixel_values = batch["pixel_values"].to(device)
                labels = batch["labels"].to(device)

                # evaluate
                with torch.no_grad():
                  # forward + backward + optimize
                  outputs = model(pixel_values=pixel_values, labels=labels)
                  loss, logits = outputs.loss, outputs.logits
                  upsampled_logits = nn.functional.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
                  predicted = upsampled_logits.argmax(dim=1)
                  
                  # note that the metric expects predictions + labels as numpy arrays
                  metric.add_batch(predictions=predicted.detach().cpu().numpy(), references=labels.detach().cpu().numpy())

          metrics = metric.compute(num_labels=len(id2label), 
                          ignore_index=255,
                          reduce_labels=False, # we've already reduced the labels before)
                        )

          print("valid Loss:", loss.item())
          print("valid Mean_iou:", metrics["mean_iou"])
          print("valid Mean accuracy:", metrics["mean_accuracy"])

        #   wandb.log({"valid loss": loss.item(), "valid Mean_iou": metrics["mean_iou"]})

          if (metrics["mean_iou"]>best_iou):
            best_iou = metrics["mean_iou"]
        #     wandb.config["best_iou"] = best_iou
            torch.save({
              'epoch': epoch,
              'model_state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
              'best_iou': best_iou
              }, sys.argv[2])
