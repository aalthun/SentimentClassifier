# Sentiment Classification with Custom Transformer Encoder and BPE Tokenizer

This repository implements a sentiment analysis system using a custom **Transformer Encoder** model and a **Byte Pair Encoding (BPE)** tokenizer. Both the model and tokenizer have been implemented from scratch, ensuring flexibility and customization. The model is designed to classify text sentiment (positive, negative, neutral) based on input data.

## Overview

This project implements a **sentiment analysis system** using a **Transformer Encoder** model that was built from scratch. The model utilizes the powerful attention mechanism of transformers, specifically the **Encoder** architecture, for efficient text classification. The model is designed to categorize input text into one of three sentiment classes: **positive**, **negative**, or **neutral**.

Additionally, the system employs a custom **BPE (Byte Pair Encoding)** tokenizer for preprocessing text before feeding it into the model. The BPE tokenizer helps manage out-of-vocabulary words and reduces the vocabulary size, enabling better handling of large and complex text datasets.

### Key Components:

* **Custom Transformer Encoder**: Built from scratch to process sequences efficiently.
* **BPE Tokenizer**: For text preprocessing and efficient tokenization.
* **Sentiment Classification**: Text is classified into sentiment labels (positive, negative, neutral).

## Features

* **Custom-built Transformer Encoder** using PyTorch.
* **Byte Pair Encoding (BPE) Tokenizer** for efficient text tokenization and preprocessing.
* **Support for multi-class sentiment classification** (positive, negative, neutral).
* **End-to-end pipeline** for training, evaluating, and testing the model.
* **Scalable architecture**: Easy to modify and extend the Transformer model.

## Technologies

This project utilizes the following technologies:

* **Python**: Programming language for implementing the entire system.
* **PyTorch**: Deep learning framework used to build the Transformer Encoder model.
* **BPE Tokenizer**: Custom implementation of the Byte Pair Encoding algorithm for efficient tokenization.
* **pandas**: For data handling and manipulation.
* **scikit-learn**: For data splitting.


**Prepare your dataset**: Make sure your data is in a format that the training and testing scripts can handle (e.g., CSV, JSON).

## Usage

Once you have set up the project and installed the dependencies, you can proceed with training or inference. Below are the steps to train the model and test it.

## Model

The **Transformer Encoder** model has been built from scratch using **PyTorch**. It utilizes the **self-attention mechanism** that allows the model to focus on different parts of the input sequence while processing it. This is particularly useful for capturing dependencies and understanding the context in longer text sequences.

### Key Components of the Model:

1. **Embedding Layer**:

   * Converts input tokens into dense vectors.

2. **Multi-Head Self-Attention**:

   * The attention mechanism allows the model to weigh the importance of different tokens in the sequence, enabling it to capture complex relationships between words.

3. **Feed-forward Neural Network**:

   * After attention, the output is passed through a feed-forward neural network for further processing.

4. **Positional Encoding**:

   * To capture the order of words in the sequence, positional encodings are added to the input embeddings.

5. **Output Layer**:

   * A fully connected layer followed by a softmax activation to classify the sentiment into one of the three categories (positive, negative, neutral).

### Model Architecture (Overview):

* **Input**: Tokenized text (converted into embeddings).
* **Encoder Layer**: Multiple stacked encoder layers with attention and feed-forward networks.
* **Output**: A probability distribution over the sentiment classes (positive, negative, neutral).

## Tokenizer

The **Byte Pair Encoding (BPE)** tokenizer is used to tokenize text into subword units. This is particularly useful for handling rare words and out-of-vocabulary tokens. The BPE algorithm merges the most frequent pairs of bytes in a text corpus gradually reducing the vocabulary size and increasing efficiency.

### Key Steps in Tokenization:

1. **Text Preprocessing**: Basic text preprocessing. 
2. **BPE Encoding**: The BPE algorithm splits words into subwords allowing the model to handle unknown words by breaking them into smaller more frequent parts.
3. **Tokenization**: The processed text is tokenized into integer indices that the model can understand.


### **Additional Notes:**

* You can further experiment with the Transformer architecture (e.g., adjusting the number of attention heads, encoder layers, etc.).
* If you have a large dataset, consider using techniques like **gradient accumulation** or **mixed precision training** to optimize training speed.
