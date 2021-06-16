OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=4 python train.py \
   --dataset_name llff \
   --root_dir ../pretrained/silica \
   --N_importance 64 --img_wh 1008 756 \
   --num_epochs 20 --batch_size 4096 \
   --optimizer adam --lr 5e-4 \
   --lr_scheduler steplr --decay_step 10 15 --decay_gamma 0.5 \
   --exp_name silica_hr_fastnerf \
   --spheric 
