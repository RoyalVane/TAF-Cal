# Test-time Fourier Style Calibration for Domain Generalization
### [Paper](https://arxiv.org/abs/2205.06427) | [IJCAI-ECAI 2022](https://ijcai-22.org/) | [SMiLe Lab](https://web.northeastern.edu/smilelab/)

[Xingchen Zhao](https://www.xingchenzhao.com/),
[Chang Liu](https://sites.google.com/view/cliu5/home/),
[Anthony Sicilia](https://anthonysicilia.github.io/),
[Seong Jae Hwang](https://micv.yonsei.ac.kr/seongjae),
[Yun Raymond Fu](http://www1.ece.neu.edu/~yunfu/)

This is the official implementation of the paper "Test-time Fourier Style Calibration for Domain Generalization". The code will cleaned up soon. Please contact zhao.xingc@northeastern.edu or create an issue if you have any questions. Thank you for your interest.

## Install Conda Environment
Please visit the [Anaconda install page](https://docs.anaconda.com/anaconda/install/index.html) if you do not already have conda installed

```shell script
conda env create --name TAF_Cal --file=environment.yml
conda activate TAF_Cal
```

## Download Dataset
```shell script
bash download_dataset.sh
```

## Training and Evaluation
The evaluation stage is included in the training code.

To train the deepall, please run:
```shell script
bash pacs_deepall.sh
```

To train TAF_Cal, please run (Comparing the results of the paper, the result produced by this code will fluctuate 0.5~0.8% up or down due to the random seed. Try multiple runs and take the average):
```shell script
bash pacs_taf_cal_train.sh
```

## Reference

If our work or code helps you, please consider to cite our paper. Thank you!

```BibTeX
@inproceedings{wang2022r2l,
  author = {Xingchen Zhao and Chang Liu and Anthony Scilia and Seong Jae Hwang and Yun Fu},
  title = {Test-time Fourier Style Calibration for Domain Generalization},
  booktitle = {IJCAI},
  year = {2022}
}
```