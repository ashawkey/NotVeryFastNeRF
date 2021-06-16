OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=9 python train.py \
   --dataset_name llff \
   --root_dir ../pretrained/glyph \
   --N_importance 64 --img_wh 456 342 \
   --num_epochs 30 --batch_size 1024 \
   --optimizer adam --lr 5e-4 \
   --lr_scheduler steplr --decay_step 10 20 --decay_gamma 0.5 \
   --exp_name glyph_fastnerf \
   --spheric 
