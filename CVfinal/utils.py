import numpy as np


def alpha_blend(input_image: np.ndarray, segmentation_mask: np.ndarray, alpha: float = 0.5):
    """Alpha Blending utility to overlay segmentation masks on input images
    Args:
        input_image: a np.ndarray with 1 or 3 channels
        segmentation_mask: a np.ndarray with 3 channels
        alpha: a float value
    """
    if len(input_image.shape) == 2:
        input_image = np.stack((input_image,) * 3, axis=-1)
    blended = input_image.astype(np.float32) * alpha + segmentation_mask.astype(np.float32) * (1 - alpha)
    blended = np.clip(blended, 0, 255)
    blended = blended.astype(np.uint8)
    return blended


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.sum = 0
        self.count = 0

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, val):
        self.sum += val
        self.count += 1

    def avg(self):
        return self.sum / self.count


if __name__ == '__main__':
    import os
    import cv2
    import matplotlib
    import matplotlib.pyplot as plt

    ''' matplotlib show '''
    # dataset_path = './dataset/public/S5/01'
    # nr_image = len([name for name in os.listdir(dataset_path) if name.endswith('.jpg')])
    # h, w = 480, 640
    # dpi = matplotlib.rcParams['figure.dpi']
    # fig = plt.figure(figsize=(w / dpi, h / dpi))
    # ax = fig.add_axes([0, 0, 1, 1])
    # for idx in range(nr_image):
    #     image_name = os.path.join(dataset_path, f'{idx}.jpg')
    #     label_name = os.path.join(dataset_path, f'{idx}.png')
    #     image = cv2.imread(image_name)
    #     label = cv2.imread(label_name)
    #     blended = alpha_blend(image, label, 0.5)
    #     ax.clear()
    #     ax.imshow(blended)
    #     ax.axis('off')
    #     plt.draw()
    #     plt.pause(0.01)
    # plt.close()

    ''' cv2 save video '''
    subject = 'S5'
    root = f'./dataset/public/{subject}'
    result_root = f'./Unet_drop_noTTA/{subject}'
    for i in range(1, 27):
        print(f'{i = }')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(f'./Unet_video/{subject}_{i:02d}.mp4', fourcc, 10.0, (640, 480))

        dataset_path = os.path.join(root, f'{i:02d}')
        result_path = os.path.join(result_root, f'{i:02d}')
        nr_image = len([name for name in os.listdir(dataset_path) if name.endswith('.jpg')])
        for idx in range(nr_image):
            image_name = os.path.join(dataset_path, f'{idx}.jpg')
            label_name = os.path.join(result_path, f'{idx}.png')
            image = cv2.imread(image_name)
            label = cv2.imread(label_name)

            red, green, blue = label[:, :, 2], label[:, :, 1], label[:, :, 0]
            mask = (red == 255) & (green == 255) & (blue == 255)
            label[:, :, :3][mask] = [0, 0, 255]

            blended = alpha_blend(image, label, 0.5)
            out.write(blended)
        
        out.release()