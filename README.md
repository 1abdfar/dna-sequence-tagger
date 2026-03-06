# dna-sequence-tagger

# DNA Language Modeling for Intron/Exon Prediction

## Overview
This project leverages **HyenaDNA**, a state-of-the-art long-range genomic foundation model, to predict RNA splicing boundaries directly from raw DNA sequences. The model performs token-level sequence classification to identify whether each nucleotide belongs to an intron or an exon.

## Technical Approach
Conventional exon-intron boundary identification relies on time-consuming experimental methods like RNA-seq. This project treats DNA as a language, utilizing deep learning to capture sequence-level features that differentiate coding from non-coding regions. 

* **Model Architecture:** Implemented a custom PyTorch `TokenSequenceTagger` on top of the pre-trained `LongSafari/hyenadna-small-32k-seqlen-hf` base model. The base encoder weights were frozen, and a custom classification head (with dropout and a linear layer) was trained for the downstream token classification task.
* **Data Processing:** Handled long genomic sequences by chunking nucleotide inputs up to 16,384 tokens, managing custom padding and label masking (`-100`) for PyTorch's CrossEntropyLoss.
* **Tech Stack:** Python, PyTorch, Hugging Face `transformers`, Pandas, NumPy.

## Dataset
The dataset consists of nucleotide sequences provided in FASTA format, alongside TSV files containing nucleotide-level binary labels (`0` = Intron, `1` = Exon). 
*(Note: Data files are excluded from this repository due to their size and course policies).*

## Evaluation
The model's performance is evaluated based on nucleotide-level accuracy. The test set labels are withheld, and predictions are submitted via Codabench for blind scoring.

## How to Run
1. Clone the repository and install dependencies.
2. Ensure the dataset (`train.tsv`, `test.tsv`, `sequences.fasta`) is located in the `data/` directory.
3. Run the Jupyter Notebook `proj2_c121.ipynb` to initiate the training loop, generate validation metrics, and output the final `predictions.tsv` file.
