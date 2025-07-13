# TCRdesign: an antigen-specific generative language model for de novo design of T-cell receptors

T cell receptors (TCR), which are heterodimers of α and β chains that recognize foreign antigens, are of great significance
to current immunotherapy. Although artificial intelligence (AI) has explosively accelerated de novo protein design, the
challenge of therapeutic TCR design has been overlooked by most researchers. Existing TCR engineering relies heavily on
isolating antigen-specific TCRs from tumor tissues, which requires a large amount of labor resources and wet experimental
verification. To mitigate this issue, we present TCRdesign, a pre-trained generative protein language model for the de
novo design of artificial TCR β-chain complementarity-determining region 3 (CDR3β) sequences conditioned on antigen-
binding specificity. In parallel, we develop a high-accuracy binding predictor (TCRBinder) that couples paired α/β
chain information with antigen sequences to assess binding specificity. Our extensive experiments demonstrate that (1)
TCRdesign surpasses state-of-the-art baselines in generating antigen-specific TCR sequences. The model leverages paired-
chain coherence to refine amino-acid level interaction patterns. (2) TCRdesign-generated TCR sequences exhibit better
antigen binding capability to diverse oncogenic hotspots compared with natural counterparts. (3) TCRdesign inherits
the intrinsic properties of large protein language models, enabling effectively identify the determinant residues in TCR-
antigen binding, which enhances its interpretability. These results highlight the significant capability of TCRdesign in
understanding and generating TCR sequences with an antigen-specific interaction pattern, charting a versatile path
toward AI-driven T-cell engineering for precision immunotherapy.
![image](https://github.com/lixiaokun2020/TCRdesign/blob/main/Framework.png)

## System requirements
This tool is supported for Linux. The tool has been tested on the following systems:

+ Ubuntu Linux release 18.04

`TCRdesign` package requires only a standard computer with enough RAM and NVIDIA GPUs to support operations:

+ CPU: 10 cores, 2.5 GHz/core
+ RAM: 40 GB
+ GPU: NVIDIA GeForce A100 GPUs

## Installation
To install the required packages for running TCRdesign, please use the following command:
```bash
conda create -n <env_name> python==3.9
conda activate <env_name>
pip install -r requirements.txt
```

## How to train and use
The training of 'TCRdesign' and 'TCRBinder' consists of three steps: first, we pre-train two language models on paired TCR beta and alpha chain sequences, respectively. Then, TCRBinder is constructed and fine-tuned using Antigen-TCR binding data. Finally, we develop TCRBinder by Roformer and ESM2 using paired data for designing and evaluating the AI-generated CDR3β. The details of each training are in the `Code/config` folder. Of note, make sure you are in the '/TCRdesign/Code' folder.


### 1. Pre-train on full-length paired-chain sequences
The MAA task is used for the self-training of α-Roformer and β-Roformer. 

The training command for β-Roformer:
```bash
python pretrain_maa_main.py --config ./config/common/pretrain_maa_common_beta.json
```
The training command for α-Roformer is:
```bash
python pretrain_maa_main.py --config ./config/common/pretrain_maa_common_alpha.json
```
After the training, the pre-trained α-Roformer and β-Roformer will be saved in the `../Result_alpha/checkpoints/BERT-Pretrain-common-MAA-NGPUs/XXXX_XXXXXX` and `../Result_beta/checkpoints/BERT-Pretrain-common-MAA-NGPUs/XXXX_XXXXXX` folder, where `XXXX_XXXXXX` is the timestamp of the training.


### 2. Training TCRBinder on Antigen-TCR binding data

Before running the affinity prediction task, please copy the **absolute path** of pre-trained α-Roformer (`../Result_alpha/checkpoints/BERT-Pretrain-common-MAA-NGPUs/XXXX_XXXXXX`) and β-Roformer (`../Result_beta/checkpoints/BERT-Pretrain-common-MAA-NGPUs/XXXX_XXXXXX`) to replace the corresponding file path in the config file `TCRBinder.json`. In detail: please replace the "beta_dir" using `../Result_beta/checkpoints/BERT-Pretrain-common-MAA-NGPUs/XXXX_XXXXXX`; replace the "alpha_dir"  using `../Result_alpha/checkpoints/BERT-Pretrain-common-MAA-NGPUs/XXXX_XXXXXX`. Besides, you should also replace the "tcr_tokenizer_dir" with the path `../Result_beta/checkpoints/BERT-Pretrain-common-MAA-NGPUs/XXXX_XXXXXX`. 

The training command for TCRBinder is:
```bash
python finetuning_er_main.py --config ./config/common/TCRBinder.json
```
The trained TCRBinder will be saved in the `../Result_TP/checkpoints/BERT-Finetunning-TCR-Binding-common-agtcr/XXXX_XXXXXX` folder.
Due to the use of ESM2 model parameters as the antigen model, there may be network errors when downloading ESM2  model parameters. Please check the network settings or try again later. You can also download files in [huggingface](https://huggingface.co/facebook/esm2_t30_150M_UR50D/tree/main) and put them in `./esm2/esm2_150m` and `./cache` for tokenizer and model (You are in the '/TCRdesign/Code' now).

### 3. Training TCRdesign on seq2seq generation
Before training, please copy the **absolute path** of pre-trained β-Roformer (`../Result_beta/checkpoints/BERT-Pretrain-common-MAA-NGPUs/XXXX_XXXXXX`) to replace the corresponding file path in the config file `seq2seq_train.json`. Specifically, please replace the "TCRBert_dir" using `../Result_beta/checkpoints/BERT-Pretrain-common-MAA-NGPUs/XXXX_XXXXXX`. Besides, you should also replace the "tcr_tokenizer_dir" and "antigen_tokenizer_dir" to `../Result_beta/checkpoints/BERT-Pretrain-common-MAA-NGPUs/XXXX_XXXXXX`. 
The training command for TCRdesign is:
```bash
python finetuning_seq2seq_main.py --config ./config/common/seq2seq_train.json
```
After the training, the trained TCRdesign will be saved in the `../Result_TCR_seq2seq/checkpoints/AGTCR-Finetuning-Seq2seq-Common/XXXX_XXXXXX/` folder.

### 4. Generate artificial TCRs
Before running the generation task, please copy the **absolute path** of TCRdesign model `../Result_TCR_seq2seq/checkpoints/AGTCR-Finetuning-Seq2seq-Common/XXXX_XXXXXX/` to "resume", copy the absolute path of `../Result_beta/checkpoints/BERT-Pretrain-common-MAA-NGPUs/XXXX_XXXXXX`to "tcr_tokenizer_dir" and "antigen_tokenizer_dir" in the config file `seq2seq_generate.json`. 

Optionally, you can customize "origin_seq", "origin_alpha", "cdr3_begin", "cdr3_end", and "use_antigen" in the config file `seq2seq_generate.json`, which represent the original beta chain, the original alpha chain, the index of the beginning and end of the cdr3 region, and the sequence of the antigen, respectively.
The generation command using TCRdesign:
```bash
python generate_tcr.py --config ./config/common/seq2seq_generate.json
```
### Expected output
The artificial TCR will be saved in the `../Result_TCR_gen/datasplit/CoV_TP-Seq2seq-Evaluate-Common/XXXX_XXXXXX/result.csv`.

The generated file contains five columns: Antigen, Generated_CDR3, Beta_Chain, Alpha_Chain. Among them, Antigen and Light_Chain do not differ from the input, Generated_CDR3 is the cdr3 region sequence generated by the model, Beta_Chain replaces the natural cdr3 region with the generated Generated_CDR3.

### 5. Evaluate artificial TCRs
After generating TCRs, TCRBinder can be used to evaluate the binding specificity of the generated TCRs. Before evaluating, please copy the **absolute path** of TCR `../Result_TP/checkpoints/BERT-Finetunning-TCR-Binding-common-agtcr/0712_200217/model_best.pth` to "discriminator_resume", replace the "beta_dir" using `../Result_beta/checkpoints/BERT-Pretrain-common-MAA-NGPUs/XXXX_XXXXXX/`, replace the "alpha_dir"  using `../Result_alpha/checkpoints/BERT-Pretrain-common-MAA-NGPUs/XXXX_XXXXXX/`, and replace "tcr_tokenizer_dir" to `../Result_beta/checkpoints/BERT-Pretrain-common-MAA-NGPUs/XXXX_XXXXXX`  in the `eval_generation.json` and change the "data_dir" to `../Result_TCR_gen/datasplit/TP-Seq2seq-Evaluate-Common/XXXX_XXXXXX/result.csv`. 

The evaluation command for TCRdesign is:
```bash
python eval_generate_seq.py --config ./config/common/eval_generation.json
```
### Expected output
The evaluation result will be saved in the `../Result_TCR_eval/datasplit/Eval-genetation/XXXX_XXXXXX/test_result.csv` 

The generated file contains four columns: beta chain, alpha chain, antigen, and y_pred refers to the evaluation results. In this case, the higher the result of y_pred, the greater the probability of binding specificity. However, in other cases, y_pred is related to the evaluation label of model training, depending on whether the training label is larger and better or smaller and better.


## Data availability
Due to the space limitation, we present part of data used in this project in the folder `ProcessedData`. Full pre-training data are available from https://opig.stats.ox.ac.uk/webapps/ots.

## Contact
If you have any questions, please contact us via email: 
- [Qiang Yang](mailto: yj219722460@163.com)
