python test_seg_pr.py --img_dir $1 --save_path ./pr_segoutput1 --model_path segformer0603_b4_best.pth
python test_seg_pr.py --img_dir $1 --save_path ./pr_segoutput2 --model_path segformer0610_b4_best.pth

python test_unet_pr.py --img_dir $1 --save_path ./pr_segoutput3

python test_conf_pr.py --img_dir $1 --save_path pr_conf_pred.json --model_path 0609-2.pth
python test_conf_pr2.py --img_dir $1 --save_path pr_conf_pred2.json --model_path 0613.pth
python fit_ellipse_pr.py ./pr_segoutput1/hidden ./pr_segoutput1_fit 
python fit_ellipse_pr.py ./pr_segoutput2/hidden ./pr_segoutput2_fit 
python ensemble_seg_pr.py ./pr_segoutput1_fit ./pr_segoutput2_fit ./pr_segoutput3 ./ensemble_private
python fit_ellipse_pr.py ./ensemble_private ./ensemble_private_fit
python write_conf_pr.py ./ensemble_private_fit pr_conf_pred.json