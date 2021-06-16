OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=7 python eval.py \
    --root_dir ../pretrained/silica \
    --dataset_name llff \
    --scene_name silica \
    --img_wh 1008 756 \
    --N_importance 64 \
    --spheric_poses \
    --ckpt_path ckpts/silica_hr_fastnerf/epoch=18.ckpt
