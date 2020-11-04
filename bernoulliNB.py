import os
import numpy as np
import pandas as pd
import re, unicodedata
import string



# libraries for sparse matrix implementation
import scipy.sparse as sp
from sklearn.utils.extmath import safe_sparse_dot


# Function "remove_non_ascii" takes care of non-ascii terms in input during pre-processing. Function taken from: https://www.kdnuggets.com/2018/03/text-data-preprocessing-walkthrough-python.html

def remove_non_ascii(words):
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return ''.join(new_words)


# Function "remove_stoppers" removes words defined as stop words from input string. It is a list of general English words. List is taken from Python's NLTK list of English stop words: https://gist.github.com/sebleier/554280

def remove_stoppers(words):
    
    stopwords = []
    f = open("stopwords.txt","r")
    for line in f:
        stopwords.append(line.strip().split()[0])

    new_words=[]
    for word in words.split():
        if word not in stopwords:
            new_words.append(word)
    
    return new_words


# Function for pre-processing of data. Adapted from what was seen during Office Hours tutorial.

def process(df, t):
    
    # remove punctuation and reduce contractions such as: don't -> dont, $L_1$-norm -> L1norm
    # taken from: https://machinelearningmastery.com/clean-text-machine-learning-python/
    table = str.maketrans('', '', string.punctuation)
    df[t] = df[t].apply(lambda x : x.translate(table))

    # put all words in lower case
    df[t] = df[t].apply(lambda x : x.lower())

    # remove leading and trailing characters
    df[t] = df[t].apply(lambda x : x.strip())
    
    # remove end-of-line
    df[t] = df[t].apply(lambda x : re.sub('\n', ' ', x))
    
    # remove digits
    df[t] = df[t].apply(lambda x :''.join([i for i in x if not i.isdigit()]))
    
    # remove non-ascii terms
    df[t] = df[t].apply(lambda x : remove_non_ascii(x))
    # remove stop words
    df[t] = df[t].apply(lambda x : remove_stoppers(x))

    return df


# Bernoulli Vectorizer: vectorizes the data and builds a vocabulary of it. Based on notes seen during Office Hours session.

class BernoulliVectorizer:
    
    def __init__(self):
        self.vocab = {}

    '''
    build_vocab takes in non-vectorized data of words and builds a vocabulary from it. It goes through the text and verifies if 
    each word is already in the vocabulary. If it is not, it adds it. Else, it skips.
    Input: pandas dataframe data
    Output: no output, function modifies self.vocab dictionary
    '''
    def build_vocab(self, data):
            
        for document in data:
            for word in document: 
                if word in self.vocab.keys():
                    self.vocab[word] += 1
                else:
                    self.vocab[word] = 1
                    
    '''
    transform takes input data and goes through each document. For each document, it goes through the vocabulary 
    and sets 1 if word is in the document, 0 if not.
    Input: pandas dataframe data
    Output: data_trf, a sparse matrix: each row corresponding to different document, each column corresponding to 
    an element in the vocabulary
    '''
    
    def transform(self, data):
        i = 0
        data_trf = []
        for document in data:

            bin_vect = np.zeros(len(self.vocab.keys()))
            for w in document: 
                if w in self.vocab.keys():
                    word_idx = list(self.vocab.keys()).index(w)
                    bin_vect[word_idx] = 1

            # sparse matrix implementation
            if i==0:
                data_trf = sp.csr_matrix(bin_vect)
            else:
                bin_vect = sp.csr_matrix(bin_vect)
                data_trf = sp.vstack([data_trf,bin_vect])
            i += 1

        return data_trf
     
    '''
    fit_transform takes input training dataset, builds vocabulary and then vectorizes data
    Input: pandas dataframe data
    Output: a sparse matrix: each row corresponding to different document, each column corresponding to 
    an element in the vocabulary
    '''
    def fit_transform(self, data):
        self.build_vocab(data)
        return self.transform(data)


# BernoulliNB class: main part of the Naive Bayes algorithm that fits to training data and returns predictions on test data. Algorithm has been adapted from the Office Hours session notes.


class BernoulliNB:
    def __init__(self, alpha):
        self.alpha = alpha
    
    '''
    encoding encodes the labels for the algorithm to understand them. The dictionary self.class_dict stores the mapping 
    between numerical code (stored as value) and category (stored as key).
    Input: pandas series with length = number of training examples
    Output: a list with length of number of classes
    '''
    
    def encoding(self,labels):
        self.class_dict = {}
        classes = np.unique(labels)
        for i,c in enumerate(classes):
            self.class_dict[c]=i

        self.class_vect=[]
        for c in labels:
            self.class_vect.append(self.class_dict[c])
        
    
    '''
     fit takes the input training data and calculates the priors (probability of each document occuring) and
     the likelihoods (given we have a document of class k, what is the probability we find word w from the vocabulary)
     Input:
         X: pandas dataframe containing training documents
         labels: pandas series containing class labels
     Output:
         no output, function calculates:
         - priors as self.counts (shape = number of classes)
         - likelihoods as self.params (shape = number of classes x number of features)
    '''
    def fit(self, X, labels):
        
        self.encoding(labels) # encode labels from strings of categories to digits 0 to 14
        y = self.class_vect
        self.n_classes = len(np.unique(y)) # number of topics
        
        # calculate priors for each topic = number of document of that class/number of documents
        self.counts = np.zeros(self.n_classes)
        for i in y: 
            self.counts[i] += 1
        self.counts /= len(y)

        
        # generate n_classes x n_features compressed sparse-row matrix to store likelihoods
        self.params = sp.csr_matrix(np.zeros((self.n_classes, X.shape[1])))
        for idx in range(X.shape[0]):
            # for each class, add the number of documents per feature from training data
            self.params[y[idx]] += X[idx] 
            
        # Laplace smoothing: add alpha to all classes to cover for unseen words in a class
        self.params = sp.csr_matrix(self.params.todense() + self.alpha) 
        
        self.class_sums = np.zeros(self.n_classes)
        for i in y:
            self.class_sums[i] += 1 # number of texts of class i
        self.class_sums += self.n_classes # Laplace
        
        # self.params contains probabilities that a word occurs in a document, given its topic is known
        self.params = self.params / self.class_sums[:, np.newaxis]

    '''
    predict computes the Bernoulli Naive Bayes probability of each class for each testing document and returns
    a list of the most likely topic for each abstract
    Input:
        X: dataframe of testing features
    Output:
        pred_classes: list of predicted classes
    '''
        
    def predict(self, X):
        neg_prob = np.log(1 - self.params)

        # Computing  neg_prob · (1 - X).T  as  ∑neg_prob - X · neg_prob
        jll = safe_sparse_dot(X, (np.log(self.params) - neg_prob).T) 

        jll += np.log(self.counts) # adding priors

        jll += neg_prob.sum(axis=1).flatten()
        predictions = np.argmax(jll, axis=1)
        
        pred_classes = []
        for p in predictions:
            pred_classes.append(list(self.class_dict.keys())[list(self.class_dict.values()).index(p)])
            
        return pred_classes
    

if __name__ == "__main__":

    train, test = pd.read_csv('train.csv'), pd.read_csv('test.csv')

    print("Pre-processing...")
    train = process(train, 'Abstract')
    test = process(test, 'Abstract')

    print("Training model...")
    B = BernoulliVectorizer()
    train_vect = B.fit_transform(train['Abstract'])
    test_vect= B.transform(test['Abstract'])

    print("Fitting the model...")
    BN = BernoulliNB(alpha=1.0)
    BN.fit(train_vect,train["Category"])

    print("Saving predictions...")
    predictions = BN.predict(test_vect)
    dat = pd.DataFrame(predictions,columns=["Category"])
    dat["Id"] = dat.index
    dat.insert(0,"Id",dat.pop("Id"))
    dat.to_csv("submission.csv",index=False)
