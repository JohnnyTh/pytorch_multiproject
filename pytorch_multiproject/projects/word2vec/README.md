#### Skip-Gram Negative Sampling for word embeddings of specific corpus
<p align="center">
  <img src="https://user-images.githubusercontent.com/39649806/73250790-44adf980-41c0-11ea-928e-c1292c596249.jpg" alt="SGNS" width="600"/>
</p>

In this project a version of Skip-Gram Negative Sampling model for obtaining word embeddings was implemented as described in the original [paper](https://arxiv.org/abs/1310.4546).

<p align="justify">Training vectorized encodings of words to predict their most likely context in a sentence is a typical task of Word2Vec machine learning models. This is done 
using shallow neural networks with one hidden layer that are able to produce the desired word vectors while being fed one-hot or integer ID-encoded input words.
The main objective of such training is, interestingly, not to perform well on the task of predicting the words' context but to learn valuable 
word embeddings that can be later reused to encode words to train other common Natural Language Processing models such as LSTMs. 

<p align="justify">The value of word embeddings lies in the ability of encoded words to correlate with each other in multidimensional vector space.
Vector encodings of words with similar meaning will be positioned or oriented similarly thus allowing the model trained on these vectors 
to draw from them additional information and generalize better.

<p align="justify"> The architectures, most commonly used to train word embeddings, are Continuous Bag of Words (CBOW) and Skip-Gram models.
In this repo, Skip-Gram Negative Sampling was used to obtain the desired embeddings. The training objective was to differentiate between the 
true context of a word in a sentence and randomly generated negative context examples.

<p align="justify">To explore the ability of word vectors to reflect the relationships between the words from a certain corpus, 
the embeddings were trained on The Lord of the Rings trilogy by J. R. R. Tolkien. Some of the selected word vectors were 
converted into 2-D representations using t-SNE and the results can be seen below.

<p align="center">
  <img src="https://user-images.githubusercontent.com/39649806/73247181-0103c180-41b9-11ea-80cd-3e098e883f00.jpg" alt="lotr_tsne" width="600" height="580"/>
</p>

<p align="justify"> In addition, cosine similarity was measured between the selected words and all the other words in vocabulary. 
The results (shown below) hinted that there is indeed a strong degree of correlation between the names of certain characters and objects.

<p align="center">
  <img src="https://user-images.githubusercontent.com/39649806/73250078-d452a880-41be-11ea-894c-78018181e7ae.jpg" alt="cos_sim" width="1200"/>
</p>
