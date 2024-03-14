# Backdoor Secrets Unveiled: Identifying Backdoor Data with Optimized Scaled Prediction Consistency (ICLR 2024)

This is an official implementation of the paper [Backdoor Secrets Unveiled: Identifying Backdoor Data with Optimized Scaled Prediction Consistency](https://openreview.net/pdf?id=1OfAO2mes1)

## Getting started

Let's start by installing all the dependencies.
Please install dependencies from `requirement.txt`.
Additionally, please install ffcv and dependencies (as per instructions from https://ffcv.io/).

The full implementation consists of various stages - which includes creating a poisoned dataset, training a model on the dataset and finally using our algorithm to identify backdoor samples.  We use FFCV (https://ffcv.io/) for our experiments, which ensures faster implementation. This requires storing our requisite dataset into a custom ``.beton`` format and also using a FFCV dataloader. We have elucidated these steps below.

## Creating a poisoned dataset

### Various backdoor attacks

The implementation of various backdoor attacks are given in the folder ``datasets``. 
Each file contains a dataset class for a backdoor attack such that we can create an indexable object.
Note that required triggers for the attacks are present in ``data/triggers/``.

### Storing as FFCV format

We will use `write_dataset_ffcv.py` to write such datasets as a ``.beton`` file in ``data``. The key arguments and their usage are listed below:

- `--dataset`
    `cifar10 | tinyimagenet |imagenet200`
- `--poison_ratio`
     This argument specifies the poison ratio of the backdoor attack. e.g. 0.1 or 0.05
- `--attack`
     `Badnet | Blend | LabelConsistent | CleanLabel | Trojan | Wanet | DFST | AdaptiveBlend`
- `--save_samples`
     `True | False` This argument determines if we want to save a number of samples after the backdoor attack 
- `--target`
      This specifies the target class index for the backdoor attack.

See ``write.sh`` for recommended usage.


## Training a model

We use `trainnew.py` to train a model using a ``.beton`` dataset created in the previous step. Key arguments and their usage are as follows:

- `--dataset`
     Please see above
- `--arch`
     `res18 | vit_tiny`
- `--poison_ratio`
     Please see above
- `--attack`
     Please see above
- `--target`
     Please see above

We note that based on the trial number of a particular setting (let's say x), the model is saved in ``Results`` in a folder called ``Trial x``.
See ``train.sh`` for recommended usage.

## Bilevel Optimization using MSPC loss

This is the core part of our contribution, using which we can identify backdoor samples from a dataset. We will use `bilevel_full.py` to implement our proposed algorithm. We describe the key arguments and their usage: 

- `--dataset`
     Please see above
- `--arch`
     Please see above
- `--batch_size`
     This specifies the batch size used in each iteration of our algorithm
- `--poison_ratio`
     Please see above
- `--attack`
     Please see above
- `--epoch_inner`
     This argument specifies the number of epochs we run the inner level optimization of our bilevel formulation
- `--outer_epoch`
     This argument specifies the number of epochs we run the our whole bilevel (both inner and outer) optimization
- `--scales`
     This is a list of integers specifying the scalar values used to multiply the input to calculate the SPC / MSPC loss
- `--trialno`
     This argument specifies the trial number where the poisoned model is stored. For example, if we want to run a second trial for a particular setting, we first train a poisoned model which is stored under ``Trial 2`` (please see above section). Then we specify ``2`` for this argument to run the bilevel optimization.
- `--target`
   Please see above.


See ``bilevel.sh`` for recommended usage.


## Baseline Defenses

Here we point to the implementations of the baseline defenses mentioned in the paper. 

- SD-FCT: https://github.com/SCLBD/Effective_backdoor_defense/
- ABL : https://github.com/bboylyg/ABL
- STRIP : https://github.com/garrisongys/STRIP

## Contributors

* [Soumyadeep Pal](https://scholar.google.ca/citations?user=c2VU-_4AAAAJ&hl=en)
* [Yuguang Yao](https://CSE.msu.edu/~yaoyugua/)



## Reference

If this code base helps you, please consider citing our paper:

```
@inproceedings{
pal2024backdoor,
title={Backdoor Secrets Unveiled: Identifying Backdoor Data with Optimized Scaled Prediction Consistency},
author={Soumyadeep Pal and Yuguang Yao and Ren Wang and Bingquan Shen and Sijia Liu},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=1OfAO2mes1}
}
```







