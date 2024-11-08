# NER-model-training
Repository fine-tuned BERT for token classification using the CoNLL-03 dataset, TensorFlow, and Hugging Face Transformers.

# CoNLL Named Entity Recognition Model Training

This repository contains code for training a BERT-based Named Entity Recognition (NER) model using the CoNLL-2003 English dataset.

## Overview

The project implements a neural network model for Named Entity Recognition using the BERT architecture. It uses the HuggingFace Transformers library and TensorFlow for model implementation and training.

## Requirements

- Python 3.x
- TensorFlow
- Transformers
- NumPy
- Pandas
- Matplotlib
- tqdm

Install dependencies using:
```bash
pip install tensorflow transformers numpy pandas matplotlib tqdm
```

## Dataset

The model is trained on the CoNLL-2003 English dataset, which should be structured in the following format:
```
-DOCSTART- -X- -X- O

EU NNP B-ORG O
Germany NNP B-LOC O
...
```

Expected directory structure:
```
input/
    conll003-englishversion/
        train.txt
        test.txt
        valid.txt
```

## Model Architecture

- Base model: `bert-base-cased`
- Task: Token Classification (NER)
- Output: Entity tags following the CoNLL-2003 schema

## Training Parameters

- Epochs: 5
- Batch Size: 8
- Learning Rate: 1e-6
- Optimizer: Adam
- Loss Function: Sparse Categorical Crossentropy

## Usage

1. Prepare your data in the CoNLL format
2. Run the training script:
```python
python conll_ner_model_training.py
```

## Model Performance

The model's performance is evaluated using:
- Loss metrics on training and validation sets
- Accuracy metrics on training and validation sets

Training progress is visualized through:
- Loss curves
- Accuracy curves

## Metrics

Loss:0.015, Accuracy:0.996

## Code Structure

- Data loading and preprocessing
- Model configuration and initialization
- Tokenization
- Training loop
- Performance visualization
- Validation



## Citation

```
@inproceedings{tjong-kim-sang-de-meulder-2003-introduction,
    title = "Introduction to the {C}o{NLL}-2003 Shared Task: Language-Independent Named Entity Recognition",
    author = "Tjong Kim Sang, Erik F. and De Meulder, Fien",
    booktitle = "Proceedings of the Seventh Conference on Natural Language Learning at {HLT}-{NAACL} 2003",
    year = "2003",
    pages = "142--147"
}
```
## Accuracy

![image](https://github.com/user-attachments/assets/4b387763-4dc5-40d8-9fe3-e24efecaa147)

## Losses

![image](https://github.com/user-attachments/assets/7747eef4-e132-4c81-afd6-af094ffef20a)


