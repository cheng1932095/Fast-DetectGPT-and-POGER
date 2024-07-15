# Fast-DetectGPT
**This code is for ICLR 2024 paper "Fast-DetectGPT: Efficient Zero-Shot Detection of Machine-Generated Text via Conditional Probability Curvature"**, where we borrow or extend some code from [DetectGPT](https://github.com/eric-mitchell/detect-gpt).

[Paper](https://arxiv.org/abs/2310.05130) 
| [LocalDemo](#local-demo)
| [OnlineDemo](http://region-9.autodl.pro:21504/)
| [OpenReview](https://openreview.net/forum?id=Bpcgcr8E8Z)

## Environment
* Python3.8
* PyTorch1.10.0
* Setup the environment:
  ```bash setup.sh```
  
(Notes: our experiments are run on 1 GPU of A100 SXM4 with 80G memory.)

## Local Demo
Please run following command locally for an interactive demo:
```
python fast-detect-gpt-main/scripts/local_infer.py
```
where the default reference model is gpt-j-6B and the default sampling model is gpt-neo-2.7B.

An example looks like:
```
Please enter your text: (Press Enter twice to start processing)
当我踏上那座山巅，视野然开朗。远处的山峦如同青翠的波浪，起伏不定地延伸到天际。天空湛蓝如洗，偶尔飘过的白云像羽毛般轻盈。阳光酒在山间，将一切都染上了温暖的金色，石头上的苔藓在光影中闪烁着微微的绿光。不远处的小溪漏潺流淌，水波粼粼，似在轻轻低语。微风吹过，带来远处松林的清香，让人感觉仿佛置身于一幅宁静而美好的画卷中。

Fast-DetectGPT criterion is 2.6002, suggesting that the text has a probability of 92% to be machine-generated.
```

## Workspace
Following folders are created for our experiments:
* ./exp_main -> experiments for 5-model generations (main.sh).

### Citation
If you find this work useful, you can cite it with the following BibTex entry:

    @inproceedings{bao2023fast,
      title={Fast-DetectGPT: Efficient Zero-Shot Detection of Machine-Generated Text via Conditional Probability Curvature},
      author={Bao, Guangsheng and Zhao, Yanbin and Teng, Zhiyang and Yang, Linyi and Zhang, Yue},
      booktitle={The Twelfth International Conference on Learning Representations},
      year={2023}
    }