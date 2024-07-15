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

# POGER

### *Ten Words Only Still Help:* Improving Black-Box AI-Generated Text Detection via Proxy-Guided Efficient Re-Sampling

[Preprint](https://arxiv.org/abs/2402.09199)

## Requirements
- Python: 3.11
- CUDA: 11.8
- Python Packages:
    ``` shell
    pip install -r POGER-main/requirements.txt
    ```
(Notes: our experiments are run on 1 GPU of A100 SXM4 with 80G memory.)
## Datasets
The binary, multiclass and OOD AIGT datasets are available at [Google Drive](https://drive.google.com/drive/folders/1xxdjZedn7le_P1HunCDF_WCuoFYI0-pz?usp=sharing).

## Run
### 1. Preprocess
> This step is optional, as processed POGER Features and POGER-Mixture Features can be downloaded at [Google Drive](https://drive.google.com/drive/folders/1xxdjZedn7le_P1HunCDF_WCuoFYI0-pz?usp=sharing).

#### Obtain POGER Features

``` shell
cd POGER-main/get_feature
export HF_TOKEN=hf_xxx        # Fill in your HuggingFace access token
export OPENAI_API_KEY=sk-xxx  # Fill in your OpenAI API key

python get_poger_feature.py \
    --n 100 \
    --k 10 \
    --delta 1.2 \
    --input ../data/train.jsonl \
    --output ./train_poger_feature.jsonl
python get_poger_feature.py \
    --n 100 \
    --k 10 \
    --delta 1.2 \
    --input ../data/test.jsonl \
    --output ./test_poger_feature.jsonl
```

#### Obtain POGER-Mixture Features
##### Inference on white-box LLMs
> This part of the code is modified from [Jihuai-wpy/SeqXGPT](https://github.com/Jihuai-wpy/SeqXGPT) under the [Apache License 2.0](https://github.com/Jihuai-wpy/SeqXGPT/blob/main/LICENSE).

``` shell
cd POGER-main/get_feature/get_true_prob

# Launch inference server
nohup python backend_api.py --model gpt2 --gpu 0 --port 6001 &
nohup python backend_api.py --model gptj --gpu 0 --port 6002 &
nohup python backend_api.py --model llama2 --gpu 1 --port 6003 &
nohup python backend_api.py --model alpaca --gpu 2 --port 6004 &
nohup python backend_api.py --model vicuna --gpu 3 --port 6005 &

# Get true probability
python get_true_prob.py
```

##### Mixing true probability and estimated probability

``` shell
cd POGER-main/get_feature

python get_poger_mix_feature.py \
    --poger-feature ./train_poger_feature.jsonl \
    --true-prob ./get_true_prob/result/train_true_prob.jsonl \
    --output ./train_poger_mix_feature.jsonl
python get_poger_mix_feature.py \
    --poger-feature ./test_poger_feature.jsonl \
    --true-prob ./get_true_prob/result/test_true_prob.jsonl \
    --output ./test_poger_mix_feature.jsonl
```

### 2. Train
``` shell
cd POGER-main/POGER

# POGER
python main.py \
    --cuda \
    --model poger \
    --data-dir ../get_feature \
    --data-name full_data

## POGER-Mixture
python main.py \
    --cuda \
    --model poger_mix \
    --data-dir ../get_feature \
    --data-name full_data
```

### 3. Test
``` shell
cd POGER-main/POGER

# POGER
python main.py \
    --cuda \
    --model poger \
    --data-dir ../get_feature \
    --test ./params/params_poger_full_data.pt

# POGER-Mixture
python main.py \
    --cuda \
    --model poger_mix \
    --data-dir ../get_feature \
    --test ./params/params_poger_mix_full_data.pt
```

## How to Cite
```
@article{shi2024ten,
  title={{Ten Words Only Still Help: Improving Black-Box AI-Generated Text Detection via Proxy-Guided Efficient Re-Sampling}},
  author={Shi, Yuhui and Sheng, Qiang and Cao, Juan and Mi, Hao and Hu, Beizhe and Wang, Danding},
  journal={arXiv preprint arXiv:2402.09199},
  url={https://arxiv.org/abs/2402.09199},
  year={2024}
}
```
