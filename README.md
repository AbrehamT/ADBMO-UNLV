# ADBMO-UNLV: Alzheimer's Disease Biomarker Research

## Overview

This project focuses on using natural language processing (NLP) and machine learning techniques to analyze research papers and identify relevant biomarkers for Alzheimer's Disease (AD). The system helps researchers filter and classify scientific literature to identify papers that contain information about protein biomarkers, specifically focusing on fluid biomarkers for AD detection and monitoring.

## Project Components

### Classification System

The project implements a binary classification system for research articles that determines whether a paper is relevant to Alzheimer's Disease biomarker research. The classification is based on six primary criteria:

1. Is the paper an original research article?
2. Does the paper have Alzheimer's Disease (AD) as its main focus?
3. Does the study have sufficient sample size (n ≥ 50 for human studies)?
4. Does the paper focus on protein biomarkers (amyloid, tau, beta-amyloid)?
5. Does the research use appropriate biological fluid models?
6. Does the paper focus on blood biomarkers for AD?

### Model Training

The project includes T5 model training components for natural language processing tasks:
- Data processing utilities (`t5_training/data_fetcher.py`, `t5_training/dataset_utils.py`)
- Training scripts (`t5_training/trainer.py`, `t5_training/dist_trainer.py`)
- Experimental notebooks for testing and training (`t5_training/testing.ipynb`, `t5_training/trainer.ipynb`)

### SentrySys Experiments

The `SentrySys_Experiments` directory contains implementations for the classification system:
- `SentryLLM.py`: A Python module implementing interfaces for large language models
- `SentryLLM.ipynb`: Notebook for running experiments using the SentryLLM system
- Test data and classification results for model evaluation

## Model Customization

The project leverages several language models with custom prompting for improved classification:
- Custom model definitions in the `Modelfile`
- Various prompt templates (`prompts` directory) for different classification approaches
- Chain-of-thought (CoT) prompting strategies to improve classification accuracy
- Adapter-based fine-tuning approaches for parameter-efficient transfer learning

## Research Focus

The research is informed by several key papers in the domain of biomedical NLP and parameter-efficient fine-tuning:
- Parameter-Efficient Transfer Learning for NLP
- Adapters: A Unified Library for Parameter-Efficient and Modular Transfer Learning
- PMC-LLaMA: Towards Building Open-source Language Models for Medicine
- Domain-Specific Language Model Pretraining for Biomedical Natural Language Processing
- BioBERT: a pre-trained biomedical language representation model for biomedical text mining
- The Impact of LoRA Adapters for LLMs on Clinical NLP Classification Under Data Limitations

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch
- Transformers library
- HuggingFace account with API token (set in `.env` file)

### Installation

1. Clone this repository
2. Create a virtual environment:
   ```bash
   python -m venv adbmo_venv
   source adbmo_venv/bin/activate  # On Linux/Mac
   # or
   adbmo_venv\Scripts\activate  # On Windows
   ```
3. Install required packages (requirements file to be added)
4. Set up your HuggingFace API token in the `.env` file

### Usage

1. For classification experiments, use the notebooks in the `SentrySys_Experiments` directory
2. For model training, use the training scripts in the `t5_training` directory
3. Sample test articles are provided in the root directory:
   - `test_positive_article.txt`
   - `test_negative_article.txt`

## Contributors

Research team at the University of Nevada, Las Vegas (UNLV)

## License

[License information to be added] 
