OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=7 python eval.py \
    --root_dir ../pretrained/firekeeper \
    --dataset_name llff \
    --scene_name firekeeper_hr \
    --img_wh 912 684 \
    --N_importance 64 \
    --spheric_poses \
    --ckpt_path ckpts/firekeeper_hr_fastnerf/epoch=12.ckpt
