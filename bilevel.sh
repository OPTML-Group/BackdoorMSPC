poison_ratio=0.1
attack='Badnet'
tau=0.1
dataset='cifar10'

for trialno in  1 # 2  3
do
    python bilevel_full.py --poison_ratio $poison_ratio --attack $attack --trialno $trialno --tau $tau --dataset $dataset
done


## Please take a look at Appendix D of the paper for setting --epoch_inner --outer_epoch
## Additionally, lower epochs e.g. 2  also work well


