# LisGAN
LisGAN, Leveraging the Invariant Side of Generative Zero-Shot Learning, CVPR 2019, the pdf can be found [Here](https://arxiv.org/pdf/1904.04092.pdf)

Just <b>Run LisGAN.py</b> and have fun!

If you find it is helpful, plese cite

    @inproceedings {Li19Leveraging, 	
     title = {Leveraging the Invariant Side of Generative Zero-Shot Learning}, 	
     booktitle = {IEEE Computer Vision and Pattern Recognition (CVPR)}, 	
     year = {2019}, 	
     author = {Li, Jingjing and Jing, Mengmeng and Lu, Ke and Ding, Zhengming and Zhu, Lei and Huang, Zi} 
    } 

Many Thanks!

Datasets can be downloaded [Here](http://datasets.d2.mpi-inf.mpg.de/xian/xlsa17.zip)

If you find that the results are prone to 0, please check that you run the code with **pytorch 0.3.1** or change the accuracy calculation code from ***torch.sum(....)***  to  ***torch.sum(....).float()***


Here are some samples for your reference:

    # zsl_cub
    CUDA_VISIBLE_DEVICES=0 python3 ../LisGAN.py --proto_param1 1e-2 --proto_param2 0.001 --ratio 0.6  --manualSeed 3483 --val_every 1 --cls_weight 0.01 --preprocessing --cuda --image_embedding res101 --class_embedding att --netG_name MLP_G --netD_name MLP_CRITIC --nepoch 70 --ngh 4096 --ndh 4096 --lr 0.0001 --classifier_lr 0.001 --lambda1 10 --critic_iter 5 --dataset CUB --batch_size 64 --nz 312 --attSize 312 --resSize 2048 --syn_num 300 --outname cub 

    # zsl_flo
    CUDA_VISIBLE_DEVICES=0 python3 ../LisGAN.py --proto_param1 1e-4 --proto_param2 0.001 --ratio 0.6  --manualSeed 806 --cls_weight 0.1 --syn_num 300 --preprocessing --val_every 1 --cuda --image_embedding res101 --class_embedding att --netG_name MLP_G --netD_name MLP_CRITIC --nepoch 97 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 --dataset FLO --batch_size 64 --nz 1024 --attSize 1024 --resSize 2048 --lr 0.0001 --outname flowers

    # zsl_sun
    CUDA_VISIBLE_DEVICES=0 python3 ../LisGAN.py --proto_param1 3e-1 --proto_param2 3e-4 --ratio 0.5  --manualSeed 4115 --cls_weight 0.01 --val_every 1 --preprocessing --cuda --image_embedding res101 --class_embedding att --netG_name MLP_G --netD_name MLP_CRITIC --nepoch 54 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 --dataset SUN --batch_size 64 --nz 102 --attSize 102 --resSize 2048 --lr 0.0002 --classifier_lr 0.0005 --syn_num 100 --outname sun

    # zsl_awa
    CUDA_VISIBLE_DEVICES=0 python3 ../LisGAN.py --proto_param1 3e-2 --proto_param2 3e-5 --ratio 0.1  --manualSeed 9182 --cls_weight 0.01 --preprocessing --val_every 1 --lr 0.00001 --cuda --image_embedding res101 --class_embedding att --netG_name MLP_G --netD_name MLP_CRITIC --nepoch 30 --syn_num 300 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 --dataset AWA1 --batch_size 64 --nz 85 --attSize 85 --resSize 2048 --outname awa 

    # zsl_apy
    CUDA_VISIBLE_DEVICES=0 python3 ../LisGAN.py --proto_param1 1    --proto_param2 3e-5 --ratio 0.7  --manualSeed 9182 --cls_weight 0.01 --preprocessing --val_every 1 --lr 0.00001 --cuda --image_embedding res101 --class_embedding att --netG_name MLP_G --netD_name MLP_CRITIC --nepoch 40 --syn_num 300 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 --dataset APY --batch_size 64 --nz 64 --attSize 64 --resSize 2048 --outname apy

    # gzsl_cub
    CUDA_VISIBLE_DEVICES=0 python3 ../LisGAN.py --proto_param1 1e-2 --proto_param2 0.001 --ratio 0.2 --gzsl --manualSeed 3483 --val_every 1 --cls_weight 0.01 --preprocessing --cuda --image_embedding res101 --class_embedding att --netG_name MLP_G --netD_name MLP_CRITIC --nepoch 56 --ngh 4096 --ndh 4096 --lr 0.0001 --classifier_lr 0.001 --lambda1 10 --critic_iter 5 --dataset CUB --nclass_all 200 --batch_size 64 --nz 312 --attSize 312 --resSize 2048 --syn_num 300 --outname cub

    # gzsl_flo
    CUDA_VISIBLE_DEVICES=0 python3 ../LisGAN.py --proto_param1 1e-1 --proto_param2 3e-2 --ratio 0.4 --gzsl --nclass_all 102 --manualSeed 806 --cls_weight 0.1 --syn_num 1200 --preprocessing --val_every 1 --cuda --image_embedding res101 --class_embedding att --netG_name MLP_G --netD_name MLP_CRITIC --nepoch 80 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 --dataset FLO --batch_size 64 --nz 1024 --attSize 1024 --resSize 2048 --lr 0.0001 --classifier_lr 0.001 --outname flowers

    # gzsl_sun
    CUDA_VISIBLE_DEVICES=0 python3 ../LisGAN.py --proto_param1 3e-1 --proto_param2 3e-5 --ratio 0.1 --gzsl --manualSeed 4115 --cls_weight 0.01 --val_every 1 --preprocessing --cuda --image_embedding res101 --class_embedding att --netG_name MLP_G --netD_name MLP_CRITIC --nepoch 40 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 --dataset SUN --batch_size 64 --nz 102 --attSize 102 --resSize 2048 --lr 0.0002 --syn_num 400 --classifier_lr 0.001 --nclass_all 717 --outname sun 

    # gzsl_awa
    CUDA_VISIBLE_DEVICES=0 python3 ../LisGAN.py --proto_param1 1e-3 --proto_param2 3e-5 --ratio 0.1 --gzsl --manualSeed 9182 --cls_weight 0.01 --preprocessing --val_every 1 --lr 0.00001 --cuda --image_embedding res101 --class_embedding att --netG_name MLP_G --netD_name MLP_CRITIC --nepoch 30 --syn_num 1800 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 --nclass_all 50 --dataset AWA1 --batch_size 64 --nz 85 --attSize 85 --resSize 2048 --outname awa 

    # gzsl_apy
    CUDA_VISIBLE_DEVICES=0 python3 ../LisGAN.py --proto_param1 3e-1 --proto_param2 3e-4 --ratio 0.2 --gzsl --manualSeed 9182 --cls_weight 0.01 --preprocessing --val_every 1 --lr 0.00001 --cuda --image_embedding res101 --class_embedding att --netG_name MLP_G --netD_name MLP_CRITIC --nepoch 50 --syn_num 1800 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 --nclass_all 32 --dataset APY --batch_size 64 --nz 64 --attSize 64 --resSize 2048 --outname apy 




