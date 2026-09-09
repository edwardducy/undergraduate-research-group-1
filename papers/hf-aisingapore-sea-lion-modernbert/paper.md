# SEA-LION-ModernBERT (AI Singapore) — Archived Documentation

This file preserves the official public documentation for the SEA-LION-ModernBERT model family, archived on 2026-09-09. The model has no peer-reviewed publication. The content below is reproduced from three official source pages.

- https://docs.sea-lion.ai/models/sea-embedding/sea-modernbert
- https://huggingface.co/aisingapore/SEA-LION-ModernBERT-300M
- https://huggingface.co/aisingapore/SEA-LION-ModernBERT-600M

Site navigation and page furniture were removed. All factual model information is kept verbatim from the sources, including their original wording.

## Official Documentation (docs.sea-lion.ai)

Source: https://docs.sea-lion.ai/models/sea-embedding/sea-modernbert

Page title: "SEA-LION-ModernBERT and Embedding". Last update: 2026-03-16.

**SEA-LION** is a collection of Large Language Models (LLMs) and encoders which have been pretrained and fine-tuned for the Southeast Asia (SEA) region.

### Introduction

SEA-LION stands for *Southeast Asian Languages In One Network*.

This encoder-only model leverages the advanced **ModernBERT** architecture combined with the Gemma 3 SentencePiece tokenizer. The adoption of the **Gemma 3 tokenizer** with ModernBERT allows the model to achieve highly efficient and culturally nuanced text processing. This combination significantly improves the tokenization fertility and compression rates for complex regional scripts and diverse Southeast Asian languages, enabling the model to handle longer context windows and cross-lingual tasks with greater computational efficiency.

To achieve this level of performance, the model was developed through a rigorous, multi-stage training pipeline. The foundation was established through extensive **pre-training on 2 Trillion (2T) tokens**, followed by a mid-training phase on an additional 1 Trillion (1T) tokens. Both of these massive training phases comprehensively covered code alongside 13 specific languages: Burmese, Chinese, English, Filipino, Indonesian, Javanese, Khmer, Lao, Malay, Sundanese, Tamil, Thai, and Vietnamese.

### Model Details

#### Model Description

The SEA-LION-ModernBERT-based models are built on the ModernBERT architecture and has a vocabulary size of 262K.

For tokenization, the model employs our custom [Gemma3](https://storage.googleapis.com/deepmind-media/gemma/Gemma3Report.pdf) tokenizer, which has excellent performance for SEA languages, ensuring optimal model performance.

- **Developed by:** AI Products Pillar, AI Singapore
- **Funded by:** Singapore NRF
- **Shared by:** AI Products Pillar, AI Singapore
- **Model type:** Encoder
- **Context length:** 8k
- **Languages:** Burmese, Chinese, English, Filipino, Indonesian, Javanese, Khmer, Lao, Malay, Sundanese, Tamil, Thai, and Vietnamese
- **License:** [MIT](https://tlo.mit.edu/understand-ip/exploring-mit-open-source-license-comprehensive-guide)

#### Model Sources

- **Repository:** The weights for this model and its various training stages are being released to support transparency, research, and diverse downstream applications. [**Link to HF Repo**](https://huggingface.co/collections/aisingapore/sea-lion-modernbert-and-embedding)

### Uses

This model card details one of the variants available within this ModernBERT-based collection.

| Model Variant | Model Repository | Suggesting Applications & Use Cases |
| --- | --- | --- |
| **Fine-tuned Embedding Models** | - [aisingapore/SEA-LION-E5-Embedding-600M](https://huggingface.co/aisingapore/SEA-LION-E5-Embedding-600M) - [aisingapore/SEA-LION-ModernBERT-Embedding-300M](https://huggingface.co/aisingapore/SEA-LION-ModernBERT-Embedding-300M) - [aisingapore/SEA-LION-ModernBERT-Embedding-600M](https://huggingface.co/aisingapore/SEA-LION-ModernBERT-Embedding-600M) | - Retrieval-Augmented Generation (RAG) - Information retrieval, and search - Similarity comparisons |
| **Pre-trained Encoder Models** | - [aisingapore/SEA-LION-ModernBERT-300M](https://huggingface.co/aisingapore/SEA-LION-ModernBERT-300M) - [aisingapore/SEA-LION-ModernBERT-600M](https://huggingface.co/aisingapore/SEA-LION-ModernBERT-600M) | - Fill mask - Text classification - Fine-tuning for downstream tasks (e.g., sentiment analysis, classification). |
| **Pre-trained Model Checkpoints** | - [aisingapore/SEA-LION-ModernBERT-300M-checkpoints](https://huggingface.co/aisingapore/SEA-LION-ModernBERT-300M-checkpoints) - [aisingapore/SEA-LION-ModernBERT-600M-checkpoints](https://huggingface.co/aisingapore/SEA-LION-ModernBERT-600M-checkpoints) | - Continued Pre-Training (CPT) - Fine-tuning for downstream tasks (e.g., sentiment analysis, classification). |

The checkpoints repository contains available of model variants.

| Model Variant | Suggesting Applications & Use Cases |
| --- | --- |
| stage1-pre-training/SEA-LION-PT-300M.pt | Composer checkpoint from the **Pre-Training Stage** suitable for continued pre-training (CPT). |
| stage1-pre-training/SEA-LION-PT-300M | Folder for the HuggingFace checkpoints from the **Pre-Training Stage**, suitable for continued pre-training or fine tuning. |
| stage2-mid-training/SEA-LION-MT-300M-w-decay.pt | Composer checkpoint from the **Mid-Training stage** with learning rate annealing suitable for fine tuning with learning rate warmup. |
| stage2-mid-training/SEA-LION-MT-300M-wo-decay.pt | Composer checkpoint from the **Mid-Training stage** without learning rate annealing suitable for continued pre-training (CPT) and fine tuning without learning rate warmup. |
| stage2-mid-training/SEA-LION-MT-300M-wo-decay | Folder for the HuggingFace checkpoints from the **Mid-Training stage** without learning rate annealing. suitable for continued pre-training (CPT) and fine tuning without learning rate warmup. |

Note: For stage2-mid-train*ing checkpoints with learning rate annealing, please refer to* [*aisingapore/SEA-LION-ModernBERT-300M*](https://huggingface.co/aisingapore/SEA-LION-ModernBERT-300M) and [*aisingapore/SEA-LION-ModernBERT-600M*](https://huggingface.co/aisingapore/SEA-LION-ModernBERT-600M)

*Note: If you are deploying our models for your specific use case, we would love to hear from you! Please feel free to* [*contact us*](mailto:sealion@aisingapore.org) *to share your experience or explore potential collaborations.*

#### Bias, Risks, and Limitations

The model was not tested for robustness against adversarial usage. It is important for users to be aware that our model exhibits certain limitations that warrant consideration. Users should also exercise caution in continue-implementing and validating the model's responses due to the potential inconsistencies.

#### Recommendations

Users (both direct and downstream) should be made aware of the risks, biases and limitations of the model.

### How to Get Started with the Model

Use this code snippet below to run inference with the model via the API.

```python
from openai import OpenAI

client = OpenAI(api_key="YOUR_API_KEY", base_url="https://api.sea-lion.ai/v1/embeddings")

result = client.embeddings.create(
    model="aisingapore/SEA-LION-ModernBERT-Embedding-600M",
    input=[
        "Singapore is a tropical island city-state.",
        "The Lion City sits at the tip of the Malay Peninsula.",
    ],
)

for item in result.data:
    print(f"index {item.index}: dim={len(item.embedding)}")
```

Use the code below to download the model locally.

```bash
pip install -U transformers>=4.48.0
```

```python
###########################
# Download checkpoints locally for continued pre-training or fine tuning
###########################
from huggingface_hub import snapshot_download

# Download the stage-1-pre-training Huggingface checkpoint
snapshot_download(
  "aisingapore/SEA-LION-ModernBERT-300M-checkpoints",
  repo_type="model",
  allow_patterns=["stage1-pre-training/SEA-LION-PT-300M/*"],
  local_dir="checkpoints"
)

# Download the stage-1-pre-training Composer checkpoint
snapshot_download(
  "aisingapore/SEA-LION-ModernBERT-300M-checkpoints",
  repo_type="model",
  allow_patterns=["stage1-pre-training/SEA-LION-PT-300M.pt"],
  local_dir="checkpoints"
)
```

```python
import torch
from transformers import pipeline

pipeline = pipeline(
    task="fill-mask",
    model="checkpoints/stage1-pre-training/SEA-LION-PT-300M", # loading from local folder
    dtype=torch.float16,
    device=0
)

pipeline("Plants create  through a process known as photosynthesis.")
```

*Note: To get started with Continued Pre-Training of the Composer checkpoints, we recommend refering to this* [*guide*](https://huggingface.co/blog/thomas-sounack/bioclinical-modernbert-tutorial)*.*

### Training Details

The models are pre-trained from scratch through a two-phase pipeline, beginning with an extensive initial stage on 2 trillion tokens, followed by a mid-training phase on an additional 1 trillion tokens. Both phases incorporated a diverse dataset covering programming code and 13 languages: Burmese, Chinese, English, Filipino, Indonesian, Javanese, Khmer, Lao, Malay, Sundanese, Tamil, Thai, and Vietnamese.

#### Training Data

The pre-trained checkpoints were pre-trained from scratch on a number of trillion tokens corpus with the following linguistic and thematic distribution:

| Data Source | Percentage |
| --- | --- |
| code | 10% |
| EN - English | 35% |
| ID - Indonesian | 8% |
| JV - Javanese | 0.5% |
| KM - Khmer | 1.5% |
| LO - Lao | 0.5% |
| MS - Malay | 4.75% |
| MY - Burmese | 1.75% |
| SU - Sundanese | 0.5% |
| TA - Tamil | 4.5% |
| TH - Thai | 8% |
| TL - Filipino | 2.5% |
| VI - Vietnamese | 8.5% |
| ZH - Chinese | 14% |

### Evaluation

#### Testing Data, Factors & Metrics

##### Testing Data

The model is evaluated across three primary benchmark suites to provide a comprehensive assessment of embedding quality across Southeast Asian, Chinese, and English contexts:

- [**SEA-BED (Southeast Asia Embedding Benchmark)**](https://arxiv.org/pdf/2508.12243): The primary testing suite, consisting of 169 datasets across 10 Southeast Asian languages (Burmese, Filipino, Indonesian, Khmer, Malay, Lao, Tamil, Tetum, Thai, and Vietnamese). Notably, 71% of these datasets are native-authored or human-curated to preserve regional linguistic properties.
- **CMTEB (Chinese Massive Text Embedding Benchmark)**: A specialised subset of MTEB focused on Chinese language tasks, used to evaluate performance in one of the region's most prominent scripts.
- **MTEB (Massive Text Embedding Benchmark)**: The industry-standard global benchmark used to gauge general-purpose English embedding performance across a wide array of tasks.

### Results

For details on Performance comparison of embedding models on SEA-BED, please refer to the [SEA-HELM](https://leaderboard.sea-lion.ai/embedding/SEA).

### Environmental Impact

Carbon emission was estimated using the fact sheet from TRG [Datacenters](https://www.trgdatacenters.com/resource/h200-power-consumption/).

- **Hardware Type:** Nvidia H200 140GB GPUs
- **Hours used:** 1,825 GPU hours
- **Cloud Provider:** SMC H200
- **Compute Region:** Singapore
- **Carbon Emitted:** appx. 513.27 kg CO2 e

### Technical Specifications

#### Model Architecture and Objective

SEA-LION-ModernBERT-300M is an encoder model using the ModernBERT architecture.

| Parameter | SEA-LION-ModernBERT |
| --- | --- |
| Layers | 22 |
| d_model | 768 |
| head_dim | 12 |
| Vocabulary | 262144 |
| Sequence Length | 8k |

### More Information

This is the repository for the commercial fine-tuned model. The model has *not* been aligned for safety. Developers and users should perform their own safety fine-tuning and related security measures. In no event shall the authors be held liable for any claims, damages, or other liabilities arising from the use of the released weights and codes.

For more info, please contact us at [sealion@aisingapore.org](mailto:sealion@aisingapore.org)

## Model Card: SEA-LION-ModernBERT-300M

Source: https://huggingface.co/aisingapore/SEA-LION-ModernBERT-300M

Model Card for SEA-LION-ModernBERT-300M. Last update: 2026-03-16.

Hub metadata: PyTorch, Safetensors, 13 languages, modernbert, arXiv 2508.12243, License: MIT, Model size 0.3B params, Tensor type F32.

SEA-LION is a collection of Large Language Models (LLMs) which have been pretrained and fine-tuned for the Southeast Asia (SEA) region.

This encoder-only model leverages the advanced **ModernBERT** architecture combined with the Gemma 3 SentencePiece tokenizer. The adoption of the **Gemma 3 tokenizer** with ModernBERT allows the model to achieve highly efficient and culturally nuanced text processing. This combination significantly improves the tokenization fertility and compression rates for complex regional scripts and diverse Southeast Asian languages, enabling the model to handle longer context windows and cross-lingual tasks with greater computational efficiency.

To achieve this level of performance, the model was developed through a rigorous, multi-stage training pipeline. The foundation was established through extensive **pre-training on 2 Trillion (2T) tokens**, followed by a mid-training phase on an additional 1 Trillion (1T) tokens. Both of these massive training phases comprehensively covered code alongside 13 specific languages: Burmese, Chinese, English, Filipino, Indonesian, Javanese, Khmer, Lao, Malay, Sundanese, Tamil, Thai, and Vietnamese.

### Model Details

#### Model Description

The SEA-LION-ModernBERT-300M models are built on the [ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base) architecture and has a vocabulary size of 262K.

For tokenization, the model employs our custom [Gemma3](https://storage.googleapis.com/deepmind-media/gemma/Gemma3Report.pdf) tokenizer, which has excellent performance for SEA languages, ensuring optimal model performance.

- **Developed by:** AI Products Pillar, AI Singapore
- **Funded by:** Singapore NRF
- **Shared by:** AI Products Pillar, AI Singapore
- **Model type:** Encoder
- **Context length:** 8k
- **Languages:** Burmese, Chinese, English, Filipino, Indonesian, Javanese, Khmer, Lao, Malay, Sundanese, Tamil, Thai, and Vietnamese
- **License:** [MIT](https://tlo.mit.edu/understand-ip/exploring-mit-open-source-license-comprehensive-guide)

#### Model Sources

- **Repository:** The weights for this model and its various training stages are being released to support transparency, research, and diverse downstream applications. [**Link to HF Repo**](https://huggingface.co/aisingapore/EA-LION-ModernBERT-300M)

### Uses

This model card details one of the variants available within this collection.

| Model Variant | Model Repository | Suggesting Applications & Use Cases |
| --- | --- | --- |
| **Fine-tuned Embedding Models** | - [aisingapore/SEA-LION-E5-Embedding-600M](https://huggingface.co/aisingapore/SEA-LION-E5-Embedding-600M) - [aisingapore/SEA-LION-ModernBERT-Embedding-300M](https://huggingface.co/aisingapore/SEA-LION-ModernBERT-Embedding-300M) - [aisingapore/SEA-LION-ModernBERT-Embedding-600M](https://huggingface.co/aisingapore/SEA-LION-ModernBERT-Embedding-600M) | - Retrieval-Augmented Generation (RAG) - Information retrieval, and search - Similarity comparisons |
| **Pre-trained Encoder Models** | - [aisingapore/SEA-LION-ModernBERT-300M](https://huggingface.co/aisingapore/SEA-LION-ModernBERT-300M) - [aisingapore/SEA-LION-ModernBERT-600M](https://huggingface.co/aisingapore/SEA-LION-ModernBERT-600M) | - Fill mask - Text classification - Fine-tuning for downstream tasks (e.g., sentiment analysis, classification). |
| **Pre-trained Model Checkpoints** | - [aisingapore/SEA-LION-ModernBERT-300M-checkpoints](https://huggingface.co/aisingapore/SEA-LION-ModernBERT-300M-checkpoints) - [aisingapore/SEA-LION-ModernBERT-600M-checkpoints](https://huggingface.co/aisingapore/SEA-LION-ModernBERT-600M-checkpoints) | - Continued Pre-Training (CPT) - Fine-tuning for downstream tasks (e.g., sentiment analysis, classification). |

*Note: If you are deploying our models for your specific use case, we would love to hear from you! Please feel free to [contact us](mailto:sealion@aisingapore.org) to share your experience or explore potential collaborations.*

#### Bias, Risks, and Limitations

The model was not tested for robustness against adversarial usage. It is important for users to be aware that our model exhibits certain limitations that warrant consideration. Users should also exercise caution in continue-implementing and validating the model's responses due to the potential inconsistencies.

#### Recommendations

Users (both direct and downstream) should be made aware of the risks, biases and limitations of the model.

### How to Get Started with the Model

Use the code below to download the model locally.

```bash
pip install -U transformers>=4.48.0
```

```python
import torch
from transformers import pipeline

pipeline = pipeline(
    task="fill-mask",
    model="aisingapore/SEA-LION-ModernBERT-300M",
    dtype=torch.float16,
    device=0
)
pipeline("Plants create  through a process known as photosynthesis.")
```

### Training Details

The models are pre-trained from scratch through a two-phase pipeline, beginning with an extensive initial stage on 2 trillion tokens, followed by a mid-training phase on an additional 1 trillion tokens. Both phases incorporated a diverse dataset covering programming code and 13 languages: Burmese, Chinese, English, Filipino, Indonesian, Javanese, Khmer, Lao, Malay, Sundanese, Tamil, Thai, and Vietnamese.

#### Training Data

**SEA-LION-ModernBERT-300M-checkpoints** was pre-trained from scratch on a number of trillion tokens corpus with the following linguistic and thematic distribution:

| Data Source | Percentage |
| --- | --- |
| code | 10% |
| EN - English | 35% |
| ID - Indonesian | 8% |
| JV - Javanese | 0.5% |
| KM - Khmer | 1.5% |
| LO - Lao | 0.5% |
| MS - Malay | 4.75% |
| MY - Burmese | 1.75% |
| SU - Sundanese | 0.5% |
| TA - Tamil | 4.5% |
| TH - Thai | 8% |
| TL - Filipino | 2.5% |
| VI - Vietnamese | 8.5% |
| ZH - Chinese | 14% |

### Evaluation

#### Testing Data, Factors & Metrics

##### Testing Data

The model is evaluated across three primary benchmark suites to provide a comprehensive assessment of embedding quality across Southeast Asian, Chinese, and English contexts:

- **[SEA-BED (Southeast Asia Embedding Benchmark)](https://arxiv.org/pdf/2508.12243)**: The primary testing suite, consisting of 169 datasets across 10 Southeast Asian languages (Burmese, Filipino, Indonesian, Khmer, Malay, Lao, Tamil, Tetum, Thai, and Vietnamese). Notably, 71% of these datasets are native-authored or human-curated to preserve regional linguistic properties.
- **CMTEB (Chinese Massive Text Embedding Benchmark)**: A specialised subset of MTEB focused on Chinese language tasks, used to evaluate performance in one of the region's most prominent scripts.
- **MTEB (Massive Text Embedding Benchmark)**: The industry-standard global benchmark used to gauge general-purpose English embedding performance across a wide array of tasks.

##### Factors

Evaluation factors are categorised by task type and linguistic diversity to ensure the model's "fertility" and "nuance" are captured accurately:

- **Linguistic Coverage**: Evaluation spans across 10+ languages, including complex Brahmic scripts (Burmese, Khmer, Lao, Tamil, Thai) and Latin-based SEA scripts (Indonesian, Filipino, Malay, Tetum, Vietnamese).
- **Task Modality**:
  - **Retrieval/Reranking:** Efficiency in finding relevant documents within a large corpus.
  - **Semantic Textual Similarity (STS):** Precision in sentence-level semantic alignment.
  - **Clustering & Classification:** Ability to group or categorize text based on latent semantic meaning.
  - **Summarisation & Bitext Mining:** High-level semantic matching and cross-lingual alignment.
- **Architecture Efficiency:** Performance is measured in the context of the ModernBERT architecture and Gemma 3 tokenizer to assess computational efficiency versus embedding quality.

##### Metrics

To provide a standardized view of performance, we report the following metrics across the benchmark suites:

- **Classification:** F1-score.
- **Multi-label Classification:** F1-score
- **Pair Classification:** Average Precision (AP).
- **Semantic Textual Similarity (STS):** Cosine similarity scores.
- **Clustering:** V-Measure Score.
- **Bitext Mining:** F1-score.
- **Retrieval & Reranking:** NDCG@10 (Primary) and MAP.
- **Instruction Retrieval:** NNDCG@5

### Results

Performance comparison of embedding models on SEA-BED ([https://leaderboard.sea-lion.ai/embedding/SEA](https://leaderboard.sea-lion.ai/embedding/SEA)). Captured on 13/03/2026 02:50pm. (Results image SEA-Embedding-Results.png hosted in the model repository.)

### Environmental Impact

Carbon emission was estimated using the fact sheet from TRG [Datacenters](https://www.trgdatacenters.com/resource/h200-power-consumption/).

- **Hardware Type:** Nvidia H200 140GB GPUs
- **Hours used:** 1,825 GPU hours
- **Cloud Provider:** SMC H200
- **Compute Region:** Singapore
- **Carbon Emitted:** appx. 513.27 kg CO2 e

### Technical Specifications

#### Model Architecture and Objective

SEA-LION-ModernBERT-300M is an encoder model using the ModernBERT architecture.

| Parameter | SEA-LION-ModernBERT |
| --- | --- |
| Layers | 22 |
| d_model | 768 |
| head_dim | 12 |
| Vocabulary | 262144 |
| Sequence Length | 8k |

#### Compute Infrastructure

##### Hardware

- **Hardware Type:** Nvidia H200 140GB GPUs
- **Cloud Provider:** SMC H200

##### Software

SEA-LION was trained using the [ModernBERT code base](https://github.com/AnswerDotAI/ModernBERT) which is powered by the [Composer](https://github.com/mosaicml/composer) training framework from MosaicML.

### Glossary

- **SEA-BED:** Southeast Asia Embedding Benchmark – a comprehensive evaluation suite for embedding models on SEA languages.
- **Asymmetric Retrieval:** Retrieval tasks where query and document formulations differ.
- **Mean Pooling:** Aggregating token embeddings by averaging (weighted by attention mask) to produce a fixed-size sentence representation.

### More Information

This is the repository for the commercial fine-tuned model. The model has *not* been aligned for safety. Developers and users should perform their own safety fine-tuning and related security measures. In no event shall the authors be held liable for any claims, damages, or other liabilities arising from the use of the released weights and codes.

For more info, please contact us at [sealion@aisingapore.org](mailto:sealion@aisingapore.org)

### Team

Ahmed Dabeer, Ahn Jeongmi, Antonyrex Sajeban, Chan Hok Teng Adwin, Cheng Zi Yi Nicholas, Choa Hsueh Mei Esther, Heng Jonathan, Huang Yuli, Jann Railey Estrada Montalan, Lee Chwan Ren, Leong Wai Yi, Leong Wei Qi, Liew Rachel, **Limkonchotiwat Peerat**, Muhammad Ridzuan Bin Mokhtar, Nagarajan Karthik, **Ng Boon Cheong Raymond**, Ngee Chia Tai, Ngui Jian Gang, Nguyen Thanh Ngan, Ong Tat-Wee David, Ong Zhi Hao, Pereira Mark, Poon Joseph, Rengarajan Hamsawardhini, Siow Wei Kang Bryan, Susanto Yosephine, Sutaveephamochanon Anocha, Tan Choon Meng, Tan Chor Phin Evelyn, Tan Siao Wei Jessica, Tan Yixian, Tee Jun Yun, Teng Kok Wai Walter, Teo Eng Sipp Leslie, Tjhi William, Wu Donghang, Yeo Yeow Tong, Yong Xianbin, Zhang Haoyang, Zhang Zhou

### Acknowledgement

This project is supported by the National Research Foundation Singapore and Infocomm Media Development Authority (IMDA), Singapore under its National Large Language Model Funding Initiative.

### Contact

[sealion@aisingapore.org](mailto:sealion@aisingapore.org)

### Associated collection and paper

- Collection: SEA-LION ModernBERT and Embedding (16 items, updated Jun 19): https://huggingface.co/collections/aisingapore/sea-lion-modernbert-and-embedding
- Paper listed on the model page: SEA-BED: Southeast Asia Embedding Benchmark, arXiv 2508.12243, published Aug 17, 2025.

## Model Card: SEA-LION-ModernBERT-600M

Source: https://huggingface.co/aisingapore/SEA-LION-ModernBERT-600M

Model Card for SEA-LION-ModernBERT-600M. Last update: 2026-03-16.

Hub metadata: PyTorch, Safetensors, 13 languages, modernbert, arXiv 2508.12243, License: MIT, Model size 0.6B params, Tensor type F32.

SEA-LION is a collection of Large Language Models (LLMs) which have been pretrained and fine-tuned for the Southeast Asia (SEA) region.

Leveraging the advanced encoder-only **ModernBERT** architecture combined with the **Gemma 3 SentencePiece tokenizer**, this specific model achieves highly efficient and culturally nuanced text processing with improved tokenization fertility and compression rates for complex regional scripts, enabling it to handle longer context windows and cross-lingual tasks with greater computational efficiency. It was developed through a multi-stage training pipeline, establishing its foundation through extensive **pre-training on 2 Trillion (2T) tokens**, followed by a mid-training phase on an additional 1 Trillion (1T) tokens, with both phases covering code and 13 languages: Burmese, Chinese, English, Filipino, Indonesian, Javanese, Khmer, Lao, Malay, Sundanese, Tamil, Thai, and Vietnamese. To enhance cross-lingual alignment, the model then underwent contrastive pre-training using 245 million text pairs (EN-EN and EN-SEA), and was finally instruct-tuned using a diverse dataset of 8 million text pairs (spanning EN-EN, CN-CN, EN-SEA, and SEA-SEA) to create the final instruction-tuned model.

### Model Details

#### Model Description

The SEA-LION-ModernBERT-600M model is built on the ModernBERT-large architecture and has a vocabulary size of 262K.

For tokenization, the model employs our custom [Gemma 3](https://storage.googleapis.com/deepmind-media/gemma/Gemma3Report.pdf) tokenizer, which has excellent performance for SEA languages, ensuring optimal model performance.

- **Developed by:** AI Products Pillar, AI Singapore
- **Funded by:** Singapore NRF
- **Shared by:** AI Products Pillar, AI Singapore
- **Model type:** Encoder
- **Context length:** 8k
- **Languages:** Burmese, Chinese, English, Filipino, Indonesian, Javanese, Khmer, Lao, Malay, Sundanese, Tamil, Thai, and Vietnamese
- **License:** [MIT](https://tlo.mit.edu/understand-ip/exploring-mit-open-source-license-comprehensive-guide)

#### Model Sources

- **Repository:** We are releasing the weights for every training stage to promote transparency and research and to support a wide range of downstream applications. The final SEA-LION-Embedding-600M (fine-tuned variant) can be accessed at the [**Link to HF Repo**](https://huggingface.co/aisingapore/SEA-Embedding-600M)

### Uses

This model card details one of the variants available within this collection.

| Model Variant | Model Repository | Suggesting Applications & Use Cases |
| --- | --- | --- |
| **Fine-tuned Embedding Models** | - [aisingapore/SEA-LION-E5-Embedding-600M](https://huggingface.co/aisingapore/SEA-LION-E5-Embedding-600M) - [aisingapore/SEA-LION-ModernBERT-Embedding-300M](https://huggingface.co/aisingapore/SEA-LION-ModernBERT-Embedding-300M) - [aisingapore/SEA-LION-ModernBERT-Embedding-600M](https://huggingface.co/aisingapore/SEA-LION-ModernBERT-Embedding-600M) | - Retrieval-Augmented Generation (RAG) - Information retrieval, and search - Similarity comparisons |
| **Pre-trained Encoder Models** | - [aisingapore/SEA-LION-ModernBERT-300M](https://huggingface.co/aisingapore/SEA-LION-ModernBERT-300M) - [aisingapore/SEA-LION-ModernBERT-600M](https://huggingface.co/aisingapore/SEA-LION-ModernBERT-600M) | - Fill mask - Text classification - Fine-tuning for downstream tasks (e.g., sentiment analysis, classification). |
| **Pre-trained Model Checkpoints** | - [aisingapore/SEA-LION-ModernBERT-300M-checkpoints](https://huggingface.co/aisingapore/SEA-LION-ModernBERT-300M-checkpoints) - [aisingapore/SEA-LION-ModernBERT-600M-checkpoints](https://huggingface.co/aisingapore/SEA-LION-ModernBERT-600M-checkpoints) | - Continued Pre-Training (CPT) - Fine-tuning for downstream tasks (e.g., sentiment analysis, classification). |

#### Bias, Risks, and Limitations

The model was not tested for robustness against adversarial usage. It is important for users to be aware that our model exhibits certain limitations that warrant consideration. Users should also exercise caution in continue-implementing and validating the model's responses due to the potential inconsistencies.

#### Recommendations

Users (both direct and downstream) should be made aware of the risks, biases and limitations of the model.

### How to Get Started with the Model

Use the code below to get started with the model.

```bash
pip install -U transformers>=4.48.0
```

```python
####################
# ## Example code adopted from https://huggingface.co/docs/transformers/main/en/model_doc/modernbert#modernbert
####################
import torch
from transformers import pipeline

pipeline = pipeline(
    task="fill-mask",
    model="aisingapore/SEA-LION-ModernBERT-600M"
    dtype=torch.float16,
    device=0
)
pipeline("Plants create  through a process known as photosynthesis.")
```

Use the code snippet below to run inference with the model via the API.

```python
from openai import OpenAI

client = OpenAI(api_key="YOUR_API_KEY", base_url="https://api.sea-lion.ai/v1/embeddings")

result = client.embeddings.create(
    model="aisingapore/SEA-LION-ModernBERT-Embedding-600M",
    input=[
        "Singapore is a tropical island city-state.",
        "The Lion City sits at the tip of the Malay Peninsula.",
    ],
)

for item in result.data:
    print(f"index {item.index}: dim={len(item.embedding)}")
```

### Training Details

#### Training Data

This model was tuned using a multi-stage training pipeline with the following datasets:

- **Contrastive Pre-training:** 245 million text pairs (EN-EN and EN-SEA) to enhance cross-lingual alignment.
- **Fine-tuning:** 8 million diverse text pairs (spanning EN-EN, CN-CN, EN-SEA, and SEA-SEA) to create the final fine-tuned model.

| **Language** | Percentage |
| --- | --- |
| EN-EN | 20% |
| CN-CN | 20% |
| EN-SEA | 10% |
| SEA-SEA | 50% |

#### Training Procedure

##### Preprocessing

Following the foundational training, the model's cross-lingual alignment was substantially enhanced by undergoing contrastive pre-training utilizing 245 million text pairs, specifically focusing on English-to-English and English-to-Southeast Asian language mappings (EN-EN and EN-SEA). Finally, to ensure the model could effectively follow user instructions and handle complex interactions, it was instruct-tuned using a diverse dataset of 8 million text pairs spanning EN-EN, CN-CN, EN-SEA, and SEA-SEA, culminating in the final highly capable instruction-tuned model.

### Evaluation

#### Testing Data, Factors & Metrics

##### Testing Data

The model is evaluated across three primary benchmark suites to provide a comprehensive assessment of embedding quality across Southeast Asian, Chinese, and English contexts:

- [**SEA-BED (Southeast Asia Embedding Benchmark)**](https://arxiv.org/pdf/2508.12243): The primary testing suite, consisting of 169 datasets across 10 Southeast Asian languages (Burmese, Filipino, Indonesian, Khmer, Malay, Lao, Tamil, Tetum, Thai, and Vietnamese). Notably, 71% of these datasets are native-authored or human-curated to preserve regional linguistic properties.
- **CMTEB (Chinese Massive Text Embedding Benchmark)**: A specialised subset of MTEB focused on Chinese language tasks, used to evaluate performance in one of the region's most prominent scripts.
- **MTEB (Massive Text Embedding Benchmark)**: The industry-standard global benchmark used to gauge general-purpose English embedding performance across a wide array of tasks.

##### Factors

Evaluation factors are categorised by task type and linguistic diversity to ensure the model's "fertility" and "nuance" are captured accurately:

- **Linguistic Coverage**: Evaluation spans across 10+ languages, including complex Brahmic scripts (Burmese, Khmer, Lao, Tamil, Thai) and Latin-based SEA scripts (Indonesian, Filipino, Malay, Tetum, Vietnamese).
- **Task Modality**:
  - **Retrieval/Reranking:** Efficiency in finding relevant documents within a large corpus.
  - **Semantic Textual Similarity (STS):** Precision in sentence-level semantic alignment.
  - **Clustering & Classification:** Ability to group or categorize text based on latent semantic meaning.
  - **Summarisation & Bitext Mining:** High-level semantic matching and cross-lingual alignment.
- **Architecture Efficiency:** Performance is measured in the context of the ModernBERT architecture and Gemma 3 tokenizer to assess computational efficiency versus embedding quality.

##### Metrics

To provide a standardized view of performance, we report the following metrics across the benchmark suites:

- **Classification:** F1-score.
- **Multi-label Classification:** F1-score
- **Pair Classification:** Average Precision (AP).
- **Semantic Textual Similarity (STS):** Cosine similarity scores.
- **Clustering:** V-Measure Score.
- **Bitext Mining:** F1-score.
- **Retrieval & Reranking:** NDCG@10 (Primary) and MAP.
- **Instruction Retrieval:** NNDCG@5

### Results

Performance comparison of embedding models on SEA-BED ([https://leaderboard.sea-lion.ai/embedding/SEA](https://leaderboard.sea-lion.ai/embedding/SEA)). Captured on 13/03/2026 02:50pm. (Results image SEA-Embedding-Results.png hosted in the model repository.)

### Environmental Impact

Carbon emission was estimated using the fact sheet from TRG [Datacenters](https://www.trgdatacenters.com/resource/h200-power-consumption/).

- **Hardware Type:** Nvidia H200 140GB GPUs
- **Hours used:** 14 hrs
- **Cloud Provider:** SMC H200
- **Compute Region:** Singapore
- **Carbon Emitted:** appx. 252.13 kg CO2 e

### Technical Specifications

#### Model Architecture and Objective

SEA-LION-ModernBERT is an encoder model using the ModernBERT architecture.

| Parameter | SEA-LION-ModernBERT-Embedding |
| --- | --- |
| Layers | 22 in 300M / 28 in 600M |
| d_model | 768 in 300M / 1024 in 600M |
| head_dim | 12 in 300M / 16 in 600M |
| Vocabulary | 262144 |
| Sequence Length | 128/ 8k |

#### Compute Infrastructure

##### Hardware

- **Hardware Type:** Nvidia H200 140GB GPUs
- **Cloud Provider:** SMC H200

##### Software

SEA-LION was trained using the [ModernBERT code base](https://github.com/AnswerDotAI/ModernBERT) which is powered by the [Composer](https://github.com/mosaicml/composer) training framework from MosaicML.

### Glossary

- **SEA-BED:** Southeast Asia Embedding Benchmark – a comprehensive evaluation suite for embedding models on SEA languages.
- **Asymmetric Retrieval:** Retrieval tasks where query and document formulations differ.
- **Mean Pooling:** Aggregating token embeddings by averaging (weighted by attention mask) to produce a fixed-size sentence representation.

### More Information

While this model supports masked language modeling, it is primarily optimized via contrastive fine-tuning for downstream tasks such as sequence classification, token classification, or question answering. Please note that these weights have not been specifically aligned for safety; therefore, developers should implement their own safety evaluations and security measures. The authors disclaim all liability for any claims, damages, or other liabilities arising from the use of the released code or weights.

For more info, please contact us at [sealion@aisingapore.org](mailto:sealion@aisingapore.org)

### Team

Ahmed Dabeer, Ahn Jeongmi, Antonyrex Sajeban, Chan Hok Teng Adwin, Cheng Zi Yi Nicholas, Choa Hsueh Mei Esther, Heng Jonathan, Huang Yuli, Jann Railey Estrada Montalan, Lee Chwan Ren, Leong Wai Yi, Leong Wei Qi, Liew Rachel, **Limkonchotiwat Peerat**, Muhammad Ridzuan Bin Mokhtar, Nagarajan Karthik, **Ng Boon Cheong Raymond**, Ngee Chia Tai, Ngui Jian Gang, Nguyen Thanh Ngan, Ong Tat-Wee David, Ong Zhi Hao, Pereira Mark, Poon Joseph, Rengarajan Hamsawardhini, Siow Wei Kang Bryan, Susanto Yosephine, Sutaveephamochanon Anocha, Tan Choon Meng, Tan Chor Phin Evelyn, Tan Siao Wei Jessica, Tan Yixian, Tee Jun Yun, Teng Kok Wai Walter, Teo Eng Sipp Leslie, Tjhi William, Wu Donghang, Yeo Yeow Tong, Yong Xianbin, Zhang Haoyang, Zhang Zhou

### Acknowledgement

This project is supported by the National Research Foundation Singapore and Infocomm Media Development Authority (IMDA), Singapore under its National Large Language Model Funding Initiative.

### Contact

[sealion@aisingapore.org](mailto:sealion@aisingapore.org)

### Associated collection and paper

- Collection: SEA-LION ModernBERT and Embedding (16 items, updated Jun 19): https://huggingface.co/collections/aisingapore/sea-lion-modernbert-and-embedding
- Paper listed on the model page: SEA-BED: Southeast Asia Embedding Benchmark, arXiv 2508.12243, published Aug 17, 2025.
