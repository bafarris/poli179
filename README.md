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

- We began by creating sentence transformer embeddings using “all-MiniLM-L6-v2”.
- Next, we split the embeddings and sentiment data into training and testing sets.
- Then we initiated the linear regression model.
- We then fit the regression model to the training data.
- For the continuous treatment, we predicted the sentiment based on the test data.
- After, we binarized the sentiment scores.
- We then evaluated the models through the MSE (continuous), accuracy score (binary), and F-1 score (binary).

## Evaluation

We will evaluate these methods through a couple of different metrics. We will evaluate based on MSE when applying sentiment scores as a continuous variable, and accuracy and F-1 score when binarizing the sentiment scores.

## Results

### BiLSTM Results

- MSE: Approximately 0.188
- Accuracy: Approximately 0.52
- F-1 Score: Approximately 0.684

### Sentence Transformer Results

- MSE: Approximately 0.12
- Accuracy: Approximately 0.833
- F-1 Score: Approximately 0.849

## Discussions

Overall, the BiLSTM model produced sub-par results and far inferior scores than that of previous research. However, the sentence transformers performed very well, potentially exceeding previous research methods. 

### Plots
#### Figure 1 BiLSTM: This scatter plot comparing actual and predicted sentiment scores shows little variation in the predicted scores, suggesting potential issues with the BiLSTM model.
![image](https://github.com/bafarris/speech-sentiment-bilstm/assets/155195678/e099ee6c-b620-4a2e-9251-73a057d3f347)

#### Figure 2 BiLSTM: This histogram compares the distribution of actual and predicted sentiment scores highlighting the narrow range in predicted scores, indicating the BiLSTM model's limitations.
![image](https://github.com/bafarris/speech-sentiment-bilstm/assets/155195678/ec0e9f16-3600-4423-a8f3-581b6386e65f)

#### Figure 3 BiLSTM: This confusion matrix shows many false positives, indicating the BiLSTM model often incorrectly predicts positive sentiment.
![image](https://github.com/bafarris/speech-sentiment-bilstm/assets/155195678/035b882e-d143-41c2-90e3-59bf577d5ad3)

#### Figure 4 BiLSTM: This ROC curve for binary sentiment predictions shows an area of 0.4, indicating this model performs worse than random guessing.
![image](https://github.com/bafarris/speech-sentiment-bilstm/assets/155195678/d88d5bfe-d8af-4d54-919f-0863f9683fe0)

#### Figure 5 Sentence Transformers: This confusion matrix for binarized predicted scores from sentence transformers shows mostly favorable outcomes, though there's some worry about the number of cases classified as 0.
![image](https://github.com/bafarris/speech-sentiment-bilstm/assets/155195678/c442366a-c078-4d77-84d0-a859d2a4b5c0)

#### Figure 6 Sentence Transformers: This scatter plot shows that actual and predicted sentiment scores follow a similar slope from the use of sentence transformers.
![image](https://github.com/bafarris/speech-sentiment-bilstm/assets/155195678/4aeb51a4-4dbc-4579-bb7c-8b8888554325)

### Challenges and Limitations

- Sentiment scores ranged from -1.24 to 2.36 in the dataset (not 0 to 10 as described in the journal).
  - We are concerned that this may influence method applications and accuracy of results.
  - We tried to use the MinMaxScaler to fix this issue but did not see a significant change in the accuracy score when applied.
- We do not know exactly how the previous researchers pre-processed and if they binarized their results to test for accuracy.
  - We are concerned that if there were differences in how we binarized results, the evaluation scores may have been influenced differently.

### Future Work

In the future, we recommend more applications of sentence transformers when trying to predict sentiment scores in political speech. We believe that these findings may have important impacts on predicting political speech sentiment. There may be potential applications to predicting other political events, policy-making decisions, and public perception of political speech.

## References
Aarsen, Nils Reimers, Tom. Sentence-Transformers: Multilingual Text Embeddings. 3.0.0. PyPI, https://www.SBERT.net. Accessed 29 May 2024.

“Bidirectional LSTM in NLP.” GeeksforGeeks, 8 June 2023, https://www.geeksforgeeks.org/bidirectional-lstm-in-nlp/.

Cochrane, Christopher. Ccochrane/emotionTranscripts. 2018. 20 June 2023. GitHub, https://github.com/ccochrane/emotionTranscripts.

Cochrane, C., Rheault, L., Godbout, J. F., Whyte, T., Wong, M. W. C., & Borwein, S. (2022). The Automatic Analysis of Emotion in Political Speech Based on Transcripts. 

Political Communication, 39(1), 98–121. https://doi.org/10.1080/10584609.2021.1952497 

“hansardExtractedSpeechesFull.Csv.Zip.”Dropbox, https://www.dropbox.com/s/4xzw3rscu7x7xn3/hansardExtractedSpeechesFull.csv.zip?dl=0&e=1. Accessed 13 May 2024.

“MinMaxScaler.” Scikit-Learn, https://scikit-learn/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html. Accessed 3 June 2024.

“NLTK Stop Words.” Pythonspot, 22 July 2021, https://pythonspot.com/nltk-stop-words/.

R, Srivignesh. “Sentiment Analysis Using Bidirectional Stacked LSTM.” Analytics Vidhya, 12 Aug. 2021, https://www.analyticsvidhya.com/blog/2021/08/sentiment-analysis-using-bidirectional-stacked-lstm/.

Senthil Kumar, N.K., Malarvizhi, N. “Bi-directional LSTM–CNN Combined method for Sentiment Analysis in Part of Speech Tagging (PoS).” Int J Speech Technol 23, 373–380 (2020). https://doi.org/10.1007/s10772-020-09716-9 

Sentiment Analysis: Bidirectional LSTM. https://kaggle.com/code/virajjayant/sentiment-analysis-bidirectional-lstm. Accessed 18 May 2024.

Sentiment Analysis with Bidirectional LSTM. https://kaggle.com/code/liliasimeonova/sentiment-analysis-with-bidirectional-lstm. Accessed 18 May 2024.

Team, Keras. Keras Documentation: Recurrent Layers. https://keras.io/api/layers/recurrent_layers/. Accessed 18 May 2024.

Thetechwriters. “Emotion Detection Using Bidirectional LSTM and Word2Vec.” Analytics Vidhya, 24 Oct. 2021, https://www.analyticsvidhya.com/blog/2021/10/emotion-detection-using-bidirectional-lstm-and-word2vec/.

Varma, Harshit. Hrshtv/Twitter-Sentiment-Analysis. 2020. 15 Mar. 2024. GitHub, https://github.com/hrshtv/Twitter-Sentiment-Analysis.

Xiao, Z., Liang, P. (2016). “Chinese Sentiment Analysis Using Bidirectional LSTM with Word Embedding.” In: Sun, X., Liu, A., Chao, HC., Bertino, E. (eds) Cloud Computing and Security. ICCCS 2016. Lecture Notes in Computer Science(), vol 10040. Springer, Cham. https://doi.org/10.1007/978-3-319-48674-1_53
