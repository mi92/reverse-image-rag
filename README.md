# Reverse Image Retrieval Cues Parametric Memory in Multimodal LLMs
The official GitHub repo with the code and experiment records for our paper [Reverse Image Retrieval Cues Parametric Memory in Multimodal LLMs](https://arxiv.org/abs/2405.18740).

## 🔥 Updates
* [**05/29/2024**] We release Reverse Image Retrieval Cues Parametric Memory in Multimodal LLMs. We propose to augment MLLM with reverse image retrieval (RIR). Check out the [paper](https://arxiv.org/abs/2405.18740) for this code.


## 📖 Abstract
Despite impressive advances in recent multimodal large language models (MLLMs), state-of-the-art models such as from the GPT-4 suite still struggle with knowledge-intensive tasks. To address this, we consider Reverse Image Retrieval (RIR) augmented generation, a simple yet effective strategy to augment MLLMs with web-scale reverse image search results. RIR robustly improves knowledge-intensive visual question answering (VQA) of GPT-4V by 37-43%, GPT-4 Turbo by 25-27%, and GPT-4o by 18-20% in terms of open-ended VQA evaluation metrics. To our surprise, we discover that RIR helps the model to better access its own world knowledge. Concretely, our experiments suggest that RIR augmentation helps by providing further visual and textual cues without necessarily containing the direct answer to a query. In addition, we elucidate cases in which RIR can hurt performance and conduct a human evaluation. Finally, we find that the overall advantage of using RIR makes it difficult for an agent that can choose to use RIR to perform better than an approach where RIR is the default setting.



## 🗺️ Overview

![Figure 1 in the paper](assets/fig1.png "Overview of the reverse image retrieval (RIR) augmented generation pipeline.")
*Figure 1: Overview of the reverse image retrieval (RIR) augmented generation pipeline. In this example, GPT-4V was used as the MLLM backbone. Calling RIR is as easy as running the following line of Python code:* `rir_api.query_with_image(image_url, query_text)`. *In this basic example
leveraging a rare bird species, the correct answer is contained in the search results which may not be the case for knowledge-intensive problems that go beyond the identification of the displayed object.*



## ⚡️ Quick Start
### 1. Install the dependencies.
Simply run
```
sh setup_env.sh
```

### 2. Data processing

We include our processed data and no additional processing is needed. 

The dataset files are provided in `local_data/`. They are:
- `infoseek_data.json`: 1650 data samples of 11 categories, with 150 samples in each category. This is the same sample used by [Li et al. \[2023\]](https://arxiv.org/abs/2311.07536).
- `snakeclef_data.json`: 300 data samples randomly sampled from the validation set of [SnakeCLEF 2023](https://www.imageclef.org/SnakeCLEF2023).

The images are uploaded to [an anonymous repo](https://anonymous.4open.science/r/rir_data/) and fetched for Idefics-2/sent to GPT via URLs, so no downloading/processing is needed.

### 3. Run Experiments

**Set up OpenAI API Key**
```
echo -n <your-openai-api-key> > cred.txt
```

**INFOSEEK.** 
To reproduce our Table 1, run 
```
sh experimenet/infoseek/reproduce_table_1.sh
```


**SnakeCLEF**
To reproduce our Table 2, run 

```
sh experimenet/snake/reproduce_table_2.sh
```

## 📚 Additional Resources

### 1. All Experiment Records

We provide all experiment records under `records/`, including:
- data samples used
- full input sent to model
- raw model predictions
- GPT judgments
- API logs.

### 2. Human Evaluation Files

We provide the original file and the annotations we created during human evaluation under `human_eval/`, the file content includes:
- sampled question and answer
- vanilla and RIR models' predictions
- GPT judgments for the models' predictions
- human annotation for the data samples.


## 📑 Citation
If you found our work useful for you, please consider citing our paper:
```
@misc{xu2024reverse,
      title={Reverse Image Retrieval Cues Parametric Memory in Multimodal LLMs}, 
      author={Jialiang Xu and Michael Moor and Jure Leskovec},
      year={2024},
      eprint={2405.18740},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```