import torch


def IoU(pred: torch.Tensor, mask: torch.Tensor, epsilon=1e-6):
    # pred and mask: BATCH x H x W

    intersection = (pred & mask).float().sum((1, 2))
    union = (pred | mask).float().sum((1, 2))
    iou = (intersection + epsilon) / (union + epsilon)
    
    return iou.tolist()

if __name__ == '__main__':
    pred = torch.tensor([[[0, 0], [0, 1]],
                         [[1, 1], [1, 1]]
                        ])
    
    mask = torch.tensor([[[0, 1], [0, 1]],
                         [[1, 0], [1, 1]]
                        ])
    
    print(pred.dtype)
    print(mask.dtype)
    print(IoU(pred, mask))