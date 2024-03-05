# CVfinal

## Environment
```shell
conda create --name pupil --file requirements.txt
```

## Download 
```shell
# Download the model checkpoints and training data to reproduce testing results
bash download.sh
```

## Preparing dataset
```shell
python splitdataset.py
```

## Training 
```shell
python train_seg.py  './data/training_set/' segformer0603_b4_best.pth
python train_seg2.py segformer0610_b4_best.pth
python train_conf.py './data' 0609-2.pth
python train_conf2.py './data' 0613.pth
python train_autoencoder.py
python train_unet.py
```

## Testing for public dataset
```shell
# This command will generate ./ensemble_public_fit/S5_solution folder, which is the final result
bash test_public.sh /directory/of/S5data
ex: bash test_public.sh ./dataset/public/S5 
```

## Testing for private dataset
```shell
# This command will generate ./ensemble_private_fit folder, which is the final result
bash test_private.sh /directory/of/hidden_dataset
ex: bash test_private.sh ./data/hidden
```