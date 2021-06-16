#OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=5,6,7,8 python train.py \
#   --dataset_name llff \
#   --root_dir ../pretrained/firekeeper \
#   --N_importance 64 --img_wh 912 684 \
#   --num_epochs 20 --batch_size 4096 \
#   --optimizer adam --lr 5e-4 \
#   --lr_scheduler steplr --decay_step 10 15 --decay_gamma 0.5 \
#   --exp_name firekeeper_hr_fastnerf \
#   --spheric \
#   --num_gpus 4
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=5 python train.py \
   --dataset_name llff \
   --root_dir ../pretrained/firekeeper \
   --N_importance 64 --img_wh 912 684 \
   --num_epochs 20 --batch_size 4096 \
   --optimizer adam --lr 5e-4 \
   --lr_scheduler steplr --decay_step 10 15 --decay_gamma 0.5 \
   --exp_name firekeeper_hr_fastnerf_signle_card \
   --spheric \
