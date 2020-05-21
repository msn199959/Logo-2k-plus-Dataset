# DRNA-Net

This is a PyTorch implementation of the AAAI2020 paper "Logo-2K+: A Large-Scale Logo Dataset for Scalable Logo Classification".

## Requirements

- Python >= 3
- PyTorch >= 0.4 Install PyTorch >=0.4 with GPU (code are GPU-only), refer to official website
- Install cupy, you can install via pip install cupy-cuda80 or(cupy-cuda90,cupy-cuda91, etc).
- Install other dependencies: pip install -r requirements.txt

## Datasets
Download the Logo-2K+(https://github.com/msn199959/Logo-2k-plus-Dataset) datasets and put it in the root directory. You can also try other classification datasets.

## Training on Logo-2K+ dataset:
Download the training, testing data. Since the program loading the data in ``drna_master/data`` by default, you can set the data path as following.
- cd drna_master
- mkdir data
- cd data
- ln -s $ dataset path

Then you can set some hyper-parameters in ``drna_master/config.py``.
If you want to train the DRNA-Net, just run ``python train.py``. During training, the log file and checkpoint file will be saved in ``save_dir`` directory. 
## Test the model
If you want to test the DRNA-Net, just run ``python test.py``. You need to specify the ``test_model`` in ``config.py`` to choose the checkpoint model for testing.


## Reference
If you are interested in our work and want to cite it, please acknowledge the following paper:

```
@inproceedings{Wang2020Logo2K,
author={Jing Wang, and Weiqing Min, and Sujuan Hou, and Shengnan Ma, and Yuanjie Zheng, and Haishuai Wang, and Shuqiang Jiang},
booktitle={AAAI Conference on Artificial Intelligence. Accepted},
title={{Logo-2K+:} A Large-Scale Logo Dataset for Scalable Logo Classification},
year={2020}
}
```
