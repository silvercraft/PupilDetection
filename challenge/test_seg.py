from transformers import SegformerFeatureExtractor
import torch
from torch import nn
from tqdm.notebook import tqdm
import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
from torchvision.transforms import transforms
from transformers import SegformerForSemanticSegmentation
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import albumentations as A
import argparse
from torchvision.utils import save_image
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


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
        # image_file_names = []
        # for root, dirs, files in os.walk(self.root_dir):
        #   image_file_names.extend(files)
        # for dir in os.listdir(self.root_dir):

        n_image = [os.path.join(self.root_dir, name) for name in os.listdir(
            self.root_dir) if name.endswith('.jpg')]
        self.images = sorted(n_image)

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

        for k, v in encoded_inputs.items():
            encoded_inputs[k].squeeze_()  # remove batch dimension

        return encoded_inputs, np.array(image), self.images[idx]


def ade_palette():
    return [[255, 255, 255], [0, 0, 0]]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Testing configuration.
    parser.add_argument('--img_dir', type=str, default='./challenge/combine')
    parser.add_argument('--save_path', type=str, default='./segoutput1')
    parser.add_argument(
        '--model_path', default='./model/segformer0610_b4_best.pth', type=str, help='model path.')

    config = parser.parse_args()

    os.makedirs(config.save_path, exist_ok=True)
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
    # print(id2label)
    # print(label2id)

    root_dir = config.img_dir
    feature_extractor = SegformerFeatureExtractor(reduce_labels=True)

    test_dataset = testDataset(
        root_dir=root_dir, feature_extractor=feature_extractor)
    print("Number of testing examples:", len(test_dataset))
    # print(test_dataset[0])
    test_dataloader = DataLoader(test_dataset, batch_size=1, num_workers=2)

    checkpoint = torch.load(config.model_path)
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

            upsampled_logits = nn.functional.interpolate(logits, size=(
                image.shape[1], image.shape[2]), mode="bilinear", align_corners=False)
            predicted = upsampled_logits.argmax(dim=1)

            # print(upsampled_logits.shape, predicted.shape)

            for i in range(predicted.shape[0]):
                # height, width, 3
                color_seg = np.zeros(
                    (image.shape[1], image.shape[2], 3), dtype=np.uint8)
                palette = np.array(ade_palette())
                for label, color in enumerate(palette):
                    color_seg[predicted[i] == label, :] = color

                color_seg = color_seg[..., ::-1]
                color_seg = color_seg.astype(np.uint8)
                # print(img_fn[i].split('\\')[-1])
                # plt.figure(figsize=(15, 10))
                # plt.imshow(color_seg)
                # plt.show()
                # dir_name = '/'.join(img_fn[i].split('/')[3:-1])
                #  print(dir_name)
                # os.makedirs(os.path.join(config.save_path, dir_name), exist_ok = True)
                head, tail = os.path.split(img_fn[i])
                plt.imsave(os.path.join(config.save_path, tail), color_seg)
