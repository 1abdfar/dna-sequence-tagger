# dna-sequence-tagger

# DNA Language Modeling for Intron/Exon Prediction

## Overview
[cite_start]This project leverages **HyenaDNA**[cite: 50], a state-of-the-art long-range genomic foundation model, to predict RNA splicing boundaries directly from raw DNA sequences. [cite_start]The model performs token-level sequence classification to identify whether each nucleotide belongs to an intron or an exon[cite: 15, 16].

## Technical Approach
[cite_start]Conventional exon-intron boundary identification relies on time-consuming experimental methods like RNA-seq[cite: 8, 9]. [cite_start]This project treats DNA as a language, utilizing deep learning to capture sequence-level features that differentiate coding from non-coding regions[cite: 10, 12]. 

* **Model Architecture:** Implemented a custom PyTorch `TokenSequenceTagger` on top of the pre-trained `LongSafari/hyenadna-small-32k-seqlen-hf` base model. The base encoder weights were frozen, and a custom classification head (with dropout and a linear layer) was trained for the downstream token classification task.
* **Data Processing:** Handled long genomic sequences by chunking nucleotide inputs up to 16,384 tokens, managing custom padding and label masking (`-100`) for PyTorch's CrossEntropyLoss.
* **Tech Stack:** Python, PyTorch, Hugging Face `transformers`, Pandas, NumPy.

## Dataset
[cite_start]The dataset consists of nucleotide sequences provided in FASTA format [cite: 23][cite_start], alongside TSV files containing nucleotide-level binary labels (`0` = Intron, `1` = Exon)[cite: 20]. 
*(Note: Data files are excluded from this repository due to their size and course policies).*

## Evaluation
[cite_start]The model's performance is evaluated based on nucleotide-level accuracy[cite: 45]. [cite_start]The test set labels are withheld, and predictions are submitted via Codabench for blind scoring[cite: 45, 71].

## How to Run
1. Clone the repository and install dependencies.
2. Ensure the dataset (`train.tsv`, `test.tsv`, `sequences.fasta`) is located in the `data/` directory.
3. [cite_start]Run the Jupyter Notebook `proj2_c121.ipynb` to initiate the training loop, generate validation metrics, and output the final `predictions.tsv` file[cite: 73].
