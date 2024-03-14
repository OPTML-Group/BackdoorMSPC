## cifar10
attack='Badnet' 
for runs in 1 2 3
do
    for poison_ratio in 0.1
    do
        CUDA_VISIBLE_DEVICES=1 python trainnew.py --poison_ratio $poison_ratio --attack $attack --save_samples 'False'
    done
done






## tinyimagenet
attack='Badnet'
for runs in  2
do
    for poison_ratio in 0.1
    do
        CUDA_VISIBLE_DEVICES=0  python trainnew.py --poison_ratio $poison_ratio --attack $attack --save_samples 'False' --dataset 'tinyimagenet'  --epochs 100 --lr 0.1 --decreasing_lr '40,80'
    done
done





## imagenet200
attack='Blend'
for runs in 1 2 3
do
    for poison_ratio in 0.1
    do
        CUDA_VISIBLE_DEVICES=1 python trainnew.py --poison_ratio $poison_ratio --attack $attack --save_samples 'False' --dataset 'imagenet200' --lr 0.5 --epochs 60
    done
done

