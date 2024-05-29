# Predicting Political Speech Sentiment Scores with Bi-LSTM and Sentence Transformers

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

The dataset that contains the segments of speech text and their corresponding human-assigned sentiment scores is located in this repository under data/w2vScores.csv. This is the dataset primarily used for our coding purposes. It was originally accessed through this GitHub repository: https://github.com/ccochrane/emotionTranscripts?tab=readme-ov-file

#### Dataset 2
The second dataset contains the coder’s sentiment scores and is 132 KB with 1,020 rows by 39 columns.

The observations will be the sentiment scores. We hope to compare the accuracy of the sentiment scores that our model generated to the accuracy scores from the coders from the study.

## Methodology

### Pre-Processing

- Loaded both datasets
- Calculated and added average sentiment score for each row in the human coder dataset
- Tokenized the text
- Removed non-alphanumeric words and stop words
- Replaced missing values with empty strings
- Created a tokens column with pre-processed text
- Kept tokens with at least 10 words
- Converted tokens column to sentences
- Initialized Word2Vec
- Gave each vector 300 dimensions
- Made a context window of 6 words around the target word
- Ignored words that appear less than 10 times
- Used 5 iterations of training

### Method 1

The first method used will be the Bidirectional Long Short-Term Memory (Bi-LSTM) model due to its solid performance record with text data from taking the context of text forward and backward simultaneously.

(So far we have been using a sample of the full data so that it doesn't use up too much RAM)

- Adjusted sample size to match both data frames
- Created a target variable (y) from the sampled sentiment scores
- Initialized tokenizer
- Created a token dictionary from the 'speech' column
- Converted texts to sequences of integers
- Padded sentences to max length
- Split the data into training and testing sets
- Built the Bi-LSTM model
- Complied the model
- Trained the model
- Evaluated the model with the testing data
- Examined the accuracy score of the model

### Method 2

The second method used to compare will be sentence transformers.

## Results and Findings So Far

### Bi-LSTM Results

So far, we have been unable to get an accuracy score that is not a 0. 

### Sentence Transformer Results

## Remaining Work and Challenges

- Too much RAM is being used when processing the entire dataset, so Google Colab keeps crashing.
- The corpus dataset will not download from Dropbox (the file won't unzip) so we are directly loading the file from Dropbox to Google Colab.
- There has been difficulty in splitting the testing and training sets due to an inconsistent number of samples.
- We are encountering difficulty in lining up the sentiment scores assigned by human coders with sections of the speech text (there is little information on the academic article or GitHub repository with the datasets about this)

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
