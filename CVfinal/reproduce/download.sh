# wget https://www.dropbox.com/s/91agi0xajsdnmpk/data.zip?dl=1 -O data.zip
# # wget -O segformer0603_b4_best.pth https://www.dropbox.com/s/20skc7jhwlwkn76/segformer0603_b4_best.pth?dl=1
# # wget -O segformer0610_b4_best.pth https://www.dropbox.com/s/oxgg1b3fmw58w1k/segformer0610_b4_best.pth?dl=1
# # wget -O 0609-2.pth https://www.dropbox.com/s/w0rzj54pwtf45gk/0609-2.pth?dl=1
# # wget -O 0613.pth https://www.dropbox.com/s/4v815bo7gs1tn69/0613.pth?dl=1
# unzip data.zip

wget -O ./dataset/public.zip https://www.dropbox.com/s/cirrx0i68cy4hqb/public.zip?dl=1
# wget -O ./model/AE_best.pt https://www.dropbox.com/s/w25gi7df002wsrw/AE_best.pt?dl=1
# wget -O ./model/unet_drop_best.pt https://www.dropbox.com/s/ce4hu9nj2dczvjm/unet_drop_best.pt?dl=1
unzip ./dataset/public.zip -d ./dataset
unzip ./dataset/public/S1.zip -d ./dataset/public
unzip ./dataset/public/S2.zip -d ./dataset/public
unzip ./dataset/public/S3.zip -d ./dataset/public
unzip ./dataset/public/S4.zip -d ./dataset/public
unzip ./dataset/public/S5.zip -d ./dataset/public
