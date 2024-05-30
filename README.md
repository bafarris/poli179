# Predicting Political Speech Sentiment Scores with BiLSTM and Sentence Transformers

> Keywords: BiLSTM (Bidirectional LSTM), Sentence Transformers

## Partners

- Brenna Farris
- Eden Stewart

## Introduction

In this study, we aim to examine political speech text and predict sentiment scores. We investigate how we can further explore predicting sentiment scores in this context through the use of a BiLSTM model and sentence transformers. What we find can be informative for future sentiment score predictions in political science through various machine learning models.

## Methods

- BiLSTM
- Sentence Transformers

## Hypothesis

In this study, we ask whether a BiLSTM model or sentence transformers can be more accurate at predicting sentiment scores for political speech than previously used methods. Cochrane et al. tested several methods for this goal and found that dictionaries based on word embeddings perform the best within their tested methods group (114). Though we are unsure if a BiLSTM or sentence transformers can surpass their level of accuracy, we hypothesize that utilization of the BiLSTM model will perform well with sentiment score prediction due to the structure of the BiLSTM model and how it processes text context (Thetechwriters).

## Data

The data examined originates from the article, “The Automatic Analysis of Emotion in Political Speech Based on Transcripts” by Cochrane et al. 

The main corpus dataset has 77,730,436 tokens from political speeches in the Canadian House of Commons. The speeches are from the 39th Parliament on January 29, 2006, to the 42nd Parliament on April 19, 2018. The availability of structured machine-readable Hansard from the 39th Parliament facilitated data collection. It was provided by the Canadian House of Commons. The dataset of the speech corpus is 1.29 GB and has 350,675 rows and 47 columns. This corpus dataset can be accessed through this DropBox link: https://www.dropbox.com/s/4xzw3rscu7x7xn3/hansardExtractedSpeechesFull.csv.zip?e=1&dl=0

The dataset containing all of the human coders' sentiment scores from the entirety of the Cochrane et al. study is located in this repository under data/fullCodingData.csv. It was originally accessed through this GitHub repository: https://github.com/ccochrane/emotionTranscripts?tab=readme-ov-file

The dataset that contains the segments of speech text and their corresponding human-assigned sentiment scores is located in this repository under data/w2vScores.csv. It has 1020 rows, 16 columns, and is 486 KB. This is the dataset primarily used for our coding purposes. It was originally accessed through this GitHub repository: https://github.com/ccochrane/emotionTranscripts?tab=readme-ov-file

## Methodology

### Pre-Processing

#### BiLSTM Pre-Processing
- Loading data
  - Load dataset from CSV file hosted on GitHub (contains human coder sentiment scores and corresponding segments of corpus)
- Examining data
  - Examine the first few rows of the dataset to gain a better understanding
  - Compute the range of sentiment scores in the dataset
- Text pre-processing
  - Tokenize speech text data
    - Convert text to lowercase
    - Split text into words
    - Remove stop words
- Train Word2Vec model
  - Train on tokens
  - This converts the words into numerical vectors to capture semantic values
- Prepare sequences
  - Convert tokens into sequences of integers (maximum length 100)
- Prepare sentiment scores
  - Store and scale to a range of -1.24 to 2.36 using MinMaxScaler
- Split data into testing and training sets
  - Save 20% for testing

#### Sentence Transformers Pre-Processing

### Method 1: BiLSTM

- We began by initializing the Sequential model.
- We then added an embedding layer that turns the integers into vectors.
- Next, we applied the BiLSTM layer, which uses past and future information.
- We then added a dense layer.
  - We used a linear activation function when classifying the sentiment scores as a continuous variable.
  - We used a sigmoid activation function when classifying the sentiment scores as a binary variable.
- After, we complied the model with an Adam optimizer.
  - We used MSE for continuous application.
  - We used binary cross entropy for binary application.
- We then evaluated the models through the MSE (continuous), accuracy score (binary), and F-1 score (binary).

### Method 2: Sentence Transformers

## Evaluation

We will evaluate these methods through a couple of different metrics. 

For the BiLSTM model, we will evaluate based on MSE when applying sentiment scores as a continuous variable, and accuracy and F-1 score when binarizing the sentiment scores.

## Results

### Bi-LSTM Results

- MSE: Approximately 0.188
- Accuracy: Approximately 0.52
- F-1 Score: Approximately 0.684

### Sentence Transformer Results

## Discussions

### Challenges and Limitations

- The corpus dataset will not download from Dropbox (the file won't unzip) so we are directly loading the file from Dropbox to Google Colab when needed.
- There has been difficulty in splitting the testing and training sets due to an inconsistent number of samples.
- Sentiment scores ranged from -1.24 to 2.36 in the dataset (not 0 to 10 as described in the journal).
  - We are concerned that this may influence method applications and accuracy of results.

### Future Work

## References
- Aarsen, Nils Reimers, Tom. Sentence-Transformers: Multilingual Text Embeddings. 3.0.0. PyPI, https://www.SBERT.net. Accessed 29 May 2024.
- “Bidirectional LSTM in NLP.” GeeksforGeeks, 8 June 2023, https://www.geeksforgeeks.org/bidirectional-lstm-in-nlp/.
- Cochrane, Christopher. Ccochrane/emotionTranscripts. 2018. 20 June 2023. GitHub, https://github.com/ccochrane/emotionTranscripts.
- Cochrane, C., Rheault, L., Godbout, J. F., Whyte, T., Wong, M. W. C., & Borwein, S. (2022). The Automatic Analysis of Emotion in Political Speech Based on Transcripts. Political Communication, 39(1), 98–121. https://doi.org/10.1080/10584609.2021.1952497 
- “hansardExtractedSpeechesFull.Csv.Zip.”Dropbox, https://www.dropbox.com/s/4xzw3rscu7x7xn3/hansardExtractedSpeechesFull.csv.zip?dl=0&e=1. Accessed 13 May 2024.
- R, Srivignesh. “Sentiment Analysis Using Bidirectional Stacked LSTM.” Analytics Vidhya, 12 Aug. 2021, https://www.analyticsvidhya.com/blog/2021/08/sentiment-analysis-using-bidirectional-stacked-lstm/.
- Senthil Kumar, N.K., Malarvizhi, N. Bi-directional LSTM–CNN Combined method for Sentiment Analysis in Part of Speech Tagging (PoS). Int J Speech Technol 23, 373–380 (2020). https://doi.org/10.1007/s10772-020-09716-9 
- Sentiment Analysis: Bidirectional LSTM. https://kaggle.com/code/virajjayant/sentiment-analysis-bidirectional-lstm. Accessed 18 May 2024.
- Sentiment Analysis with Bidirectional LSTM. https://kaggle.com/code/liliasimeonova/sentiment-analysis-with-bidirectional-lstm. Accessed 18 May 2024.
- Team, Keras. Keras Documentation: Recurrent Layers. https://keras.io/api/layers/recurrent_layers/. Accessed 18 May 2024.
- Thetechwriters. “Emotion Detection Using Bidirectional LSTM and Word2Vec.” Analytics Vidhya, 24 Oct. 2021, https://www.analyticsvidhya.com/blog/2021/10/emotion-detection-using-bidirectional-lstm-and-word2vec/.
- Varma, Harshit. Hrshtv/Twitter-Sentiment-Analysis. 2020. 15 Mar. 2024. GitHub, https://github.com/hrshtv/Twitter-Sentiment-Analysis.
- Xiao, Z., Liang, P. (2016). Chinese Sentiment Analysis Using Bidirectional LSTM with Word Embedding. In: Sun, X., Liu, A., Chao, HC., Bertino, E. (eds) Cloud Computing and Security. ICCCS 2016. Lecture Notes in Computer Science(), vol 10040. Springer, Cham. https://doi.org/10.1007/978-3-319-48674-1_53 
