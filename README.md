# Fake News Classification with BERT from Scratch

## Problem Statement
Implement a Fake News Classification system using a BERT model built from scratch in PyTorch. The goal is to classify news headlines as either "Fake" or "Non-Fake" using a custom BERT implementation and compare it with a fine-tuned pre-trained BERT model. The project also includes interpretability analysis using Integrated Gradients.

## Project Structure
- **Fake_News_Classification_final.ipynb**: Main notebook containing all code for BERT implementation, data processing, fine-tuning, and interpretability.
- **train_balanced.csv / test_balanced.csv**: Balanced datasets generated for training and testing.
- **bert_finetuned/**: Directory containing the fine-tuned BERT model and tokenizer.

## Main Components
1. **BERT from Scratch (PyTorch)**
   - Implements all core BERT components: Multi-Head Self-Attention, Feed-Forward Layers, Encoder Stack, Embeddings, Pooler, and Classifier.
   - Uses HuggingFace's `bert-base-uncased` tokenizer for input processing.
   - Tests the custom BERT model on a sample sentence to verify tensor shapes and data flow.

2. **Data Preparation**
   - Loads and combines multiple datasets.
   - Balances the dataset to ensure equal representation of both classes.
   - Splits the data into training and test sets with stratification.

3. **Fine-tuning Pre-trained BERT**
   - Loads `bert-base-uncased` from HuggingFace Transformers.
   - Fine-tunes the model on the balanced dataset for binary classification.
   - Reports training/validation loss and accuracy, and saves the best model.

4. **Evaluation & Visualization**
   - Evaluates the fine-tuned model on the test set.
   - Reports accuracy, precision, recall, F1-score, and confusion matrix.
   - Plots loss and accuracy curves.

5. **Interpretability (Integrated Gradients)**
   - Uses Captum to compute token-level attributions for model predictions.
   - Visualizes the most influential tokens for each prediction.

## Requirements
- Python 3.7+
- PyTorch
- Transformers (HuggingFace)
- scikit-learn
- pandas, numpy
- matplotlib, seaborn
- captum

Install dependencies (if running in a new environment):
```bash
pip install torch transformers scikit-learn pandas numpy matplotlib seaborn captum
```

## How to Run
1. Place the required data files (`all_train.tsv`, `all_test_public.tsv`, `all_validate.tsv`) in the working directory.
2. Open and run all cells in `Fake_News_Classification_final.ipynb` sequentially.
3. The notebook will generate balanced datasets, train and evaluate models, and produce interpretability plots.

## Results
- The notebook reports all key metrics and visualizations for model performance and interpretability.
- The fine-tuned BERT model and tokenizer are saved in the `bert_finetuned/` directory for future use.

## Credits
- BERT architecture and tokenizer: [HuggingFace Transformers](https://huggingface.co/transformers/)
- Integrated Gradients: [Captum](https://captum.ai/)

---
*This project demonstrates both the implementation of BERT from scratch and the practical application of transfer learning for fake news detection.*
