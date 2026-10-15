# BERT-Based News Headline Classification

## Project Overview

This project uses a fine-tuned BERT model to classify news headlines into 42 different news categories using the News Category Dataset v3. The model learns contextual representations of text to predict the most likely category for a given headline.

The project implements the complete NLP workflow, from data cleaning and tokenization to model fine-tuning, evaluation, and predictions on new headlines.

## Objectives

- Clean and explore the news headline dataset.
- Analyze category distributions and headline lengths.
- Tokenize text using the BERT tokenizer.
- Encode news categories into numerical labels.
- Fine-tune a pretrained BERT model for multi-class classification.
- Evaluate performance using multiple classification metrics.
- Generate the top three predicted categories for unseen headlines.

## Technologies Used

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- PyTorch
- Hugging Face Transformers
- Scikit-learn
- Jupyter Notebook

## Dataset

**News Category Dataset v3**

The dataset contains news headlines and their corresponding categories. The notebook removes duplicate records and retains the `headline` and `category` columns for classification.

- **Input:** News headline text
- **Target:** News category
- **Number of categories:** 42
- **Data split:** 80% training and 20% testing
- **Stratification:** Used to preserve category proportions across the train-test split.

## Project Workflow

1. **Data Cleaning:** Checked for missing values and duplicate records, removed duplicates, and reset the index.
2. **Exploratory Data Analysis:** Analyzed category frequencies and headline lengths using visualizations.
3. **Text Tokenization:** Used the `bert-base-uncased` tokenizer to convert headlines into token IDs and attention masks.
4. **Label Encoding:** Converted categorical news labels into numerical values using LabelEncoder.
5. **Dataset Preparation:** Created a custom PyTorch Dataset class and DataLoaders to process the data in mini-batches.
6. **Model Loading:** Loaded the pretrained `bert-base-uncased` model with a classification head configured for 42 classes.
7. **Fine-Tuning:** Trained the model using the AdamW optimizer and a linear learning-rate scheduler.
8. **Model Evaluation:** Evaluated predictions using accuracy, weighted precision, weighted recall, weighted F1-score, classification reports, and confusion matrices.
9. **Top-3 Prediction:** Implemented a function that returns the three most probable news categories and their confidence scores for a new headline.
10. **Model Saving:** Saved the fine-tuned model and tokenizer for subsequent use.

## Model Configuration

- **Pretrained model:** `bert-base-uncased`
- **Task:** Multi-class text classification
- **Number of output classes:** 42
- **Maximum sequence length:** 32 tokens
- **Batch size:** 16
- **Epochs:** 3
- **Optimizer:** AdamW
- **Learning rate:** 2e-5
- **Framework:** PyTorch and Hugging Face Transformers

## Evaluation Metrics

- **Accuracy:** Overall proportion of correctly classified headlines.
- **Precision:** Measures the correctness of positive predictions across categories.
- **Recall:** Measures how many actual examples are correctly identified.
- **F1-score:** Balances precision and recall.
- **Confusion Matrix:** Helps identify categories that the model frequently confuses.

## Key Features

- Fine-tuning a pretrained transformer for NLP classification.
- Handling 42 news categories in a single model.
- Batch-based training with GPU support when available.
- Top-3 category predictions with probability scores.
- Saving the model and tokenizer for reuse.

## How to Run

1. Clone the repository.
2. Install the required Python libraries.
3. Place the dataset at the path expected by the notebook.
4. Run the notebook cells in order to clean the data, train the model, and evaluate its performance.
5. Use the `predict_top3()` function to classify new headlines.

## Future Improvements

- Experiment with longer maximum sequence lengths.
- Tune the learning rate, batch size, and number of epochs.
- Compare BERT with DistilBERT or other transformer models.
- Analyze per-category performance and class imbalance.
- Build an interactive web application for real-time headline classification.
