import nltk
from nltk.corpus import twitter_samples                          
import pandas as pd
import numpy as np   
from utils import tweet_processing, frequency_builder

# Import Data
all_pos_tweets = twitter_samples.strings('positive_tweets.json')
all_neg_tweets = twitter_samples.strings('negative_tweets.json')

# Train-test split
train_pos = all_pos_tweets[:4000]
test_pos = all_pos_tweets[4000:]
train_neg = all_neg_tweets[:4000]
test_neg = all_neg_tweets[4000:]


train_X = train_pos + train_neg
test_X = test_pos + test_neg

# Creating array for all positive and negative labels
train_y = np.append(np.ones((len(train_pos), 1)), np.zeros((len(train_neg), 1)), axis=0)
test_y = np.append(np.ones((len(test_pos), 1)), np.zeros((len(test_neg), 1)), axis=0)

# Creating frequency dictionary
frequency = frequency_builder(train_X, train_y)


## Logistic Regression

def sigmoid(z):
    h = 1 / (1 + np.exp(-z))
    return h

def gradient(x, y, theta, alpha, num_iterations):
    '''
    X = feature matrix (m ,n+1)
    y = associated label
    theta = weight vector (n+1, 1)
    J = final cost
    theta = final weight vector
    '''
    m = x.shape[0]

    for i in range(0, num_iterations):
        z = np.dot(x, theta)
        h = sigmoid(z)
        J = -1 / m * (np.dot(y.transpose(), np.log(h)) + np.dot((1-y).transpose(), np.log(1-h)))
        theta = theta - (alpha/m)*np.dot(x.transpose(), (h-y))

    J = float(J)
    return J, theta

# Feature Extraction
def feature_extractor(tweet, frequency):
    word_list = tweet_processing(tweet)
    x = np.zeros((1, 3))
    #bias set to 1
    x[0, 0] = 1 

    for word in word_list:
        # positive tweet count
        x[0, 1] += frequency.get((word, 1.0), 0)
        # negative tweet count
        x[0, 2] += frequency.get((word, 0.0), 0)

    assert(x.shape == (1, 3))
    return x


# Training
X = np.zeros((len(train_X), 3))
for i in range(len(train_X)):
    X[i, :] = feature_extractor(train_X[i], frequency)

y = train_y

J, theta = gradient(X, y, np.zeros((3, 1)), 1e-9, 1500)
print(f"Cost after training :{J:.8f}")
print(f"Resulting vector of weights is {[round(w, 8) for w in np.squeeze(theta)]}")


# Prediction
def prection(tweet, frequency, theta):

   x = feature_extractor(tweet, frequency)
   y_pred = sigmoid(np.dot(x, theta))
   return y_pred


def predict_logistic_regression(x, y, freqs, theta):
    
    y_bar = []
    
    for tweet in x:
     
        y_pred = predict_tweet(tweet, freqs, theta)
        if y_pred > 0.5:
            # append 1.0 to the list
            y_bar.append(1)
        else:
            # append 0 to the list
            y_bar.append(0)
    
    # convert y into y_bar shape to compare
    accuracy = (y_bar==np.squeeze(y)).sum()/len(x)
    return accuracy


for tweet in ['I am happy', 'I am bad', 'this movie should have been great.']:
    print( '%s -> %f' % (tweet, prection(tweet, frequency, theta)))