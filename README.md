# Sentiment Classification with Custom Transformer Encoder and BPE Tokenizer

In this project, sentiment classification is performed using a custom Transformer Encoder and a custom BPE tokenizer which results in more efficient classification. The project assumes there are three classes but the number of classes can be increased or decreased according to the requirements. However if binary classification is to be performed the loss function should be changed to binary cross-entropy. Parameters can be adjusted to improve the model's accuracy and optimizing hyperparameters is necessary to achieve the best results.

## About BPE Tokenization

Byte Pair Encoding (BPE) tokenization analyzes byte pairs in UTF-8 encoded text and merges the most frequent byte pairs throughout the text. This process accelerates model training and makes it more efficient. When tokenization is used, the model learns faster and more effectively.

Without tokenization the training process becomes inefficient, and the training time increases significantly. Therefore applying tokenization properly is crucial for optimizing the training process.
