OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=9 python eval.py \
    --root_dir ../pretrained/glyph \
    --dataset_name llff \
    --scene_name glyph \
    --img_wh 456 342 \
    --N_importance 64 \
    --spheric_poses \
    --ckpt_path ckpts/glyph_fastnerf/epoch=25.ckpt
