# python test_seg.py --img_dir $1 --save_path ./segoutput1 --model_path segformer0603_b4_best.pth
# python test_seg.py --img_dir $1 --save_path ./segoutput2 --model_path segformer0610_b4_best.pth

# python test_unet.py ./segoutput3

# python test_conf.py --img_dir $1 --save_path conf_pred.json --model_path 0609-2.pth

python fit_ellipse.py ./segoutput1 ./segoutput1_fit S5
python fit_ellipse.py ./segoutput2 ./segoutput2_fit S5
python fit_ellipse.py ./segoutput3 ./segoutput3_fit S5
python ensemble_seg.py ./segoutput1_fit/S5 ./segoutput2_fit/S5 ./segoutput3_fit/S5 ./ensemble_public/S5_solution
python fit_ellipse.py ./ensemble_public ./ensemble_public_fit S5_solution
# python write_conf.py ./ensemble_public_fit/S5_solution conf_pred.json