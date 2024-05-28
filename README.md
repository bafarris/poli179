# Predicting Political Speech Sentiment Scores with Bi-LSTM and Sentence Transformers

## Partners

- Brenna Farris
- Eden Stewart

## Overview

In our project, we examine political speech text and aim to predict sentiment scores.

### Research Question

Is using a Bi-LSTM model more accurate for analyzing sentiment contained in political speeches than sentiment dictionaries based on word embeddings?

### Data

The data is examined in the article, “The Automatic Analysis of Emotion in Political Speech Based on Transcripts” by Cochrane et al. The main corpus dataset will be extracted from the Dropbox link in the references section. Other datasets (including coders’ sentiment scores) will be in this GitHub Repository “data” folder

#### Dataset 1
The main corpus dataset has 77,730,436 tokens from speeches in the Canadian House of Commons. The speeches are from the 39th Parliament on January 29, 2006, to the 42nd Parliament on April 19, 2018.

The availability of structured machine-readable Hansard from the 39th Parliament facilitated data collection. It was provided by the Canadian House of Commons. The dataset of the speech corpus is 1.29 GB and has 350,675 rows and 47 columns. 

#### Dataset 2
The second dataset contains the coder’s sentiment scores and is 132 KB with 1,020 rows by 39 columns.

The observations will be the sentiment scores. We hope to compare the accuracy of the sentiment scores that our model generated to the accuracy scores from the coders from the study.

## Methodology

### Pre-Processing

- 

### Method 1

The first method used will be the Bidirectional Long Short-Term Memory (Bi-LSTM) model due to its solid performance record with text data from taking the context of text forward and backward at the same time.

### Method 2

The second method used to compare will be sentence transformers.

## Results and Findings So Far

### Bi-LSTM Results

So far, we have been unable to get an accuracy score that is not a 0. 

### Sentence Transformer Results

## Remaining Work and Challenges

- Too much RAM is being used when processing the entire dataset, so Google Colab keeps crashing.
- The corpus dataset will not download from Dropbox (file won't unzip) so we are directly loading the file from Dropbox to Google Colab.
- There has been difficulty in splitting the testing and training sets due to an inconsistent number of samples.
- We are encountering difficulty in lining up the sentiment scores assigned by human coders with sections of the speech text (there is little information on the academic article or Github repository with the datasets about this)

## References
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
