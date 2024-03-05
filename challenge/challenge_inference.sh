python test_seg.py --img_dir ./challenge/$1 --save_path seg_results/$1
python test_conf.py --img_dir ./challenge --save_dir ./conf_results --target_folder $1
python write_conf.py ./conf_results/$1 ./conf_results/$1/$1.json