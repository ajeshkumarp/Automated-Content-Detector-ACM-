import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from gensim.test.utils import common_texts
from gensim.models import Word2Vec
import multiprocessing
import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
import tensorflow_hub as hub
import pandas as pd
import re
from tqdm import tqdm
from IPython.display import HTML
import yaml
import tweepy

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')



embed = hub.load("https://tfhub.dev/google/Wiki-words-250/2")

slang_dict=pd.read_csv('slangDict.csv')
slang_dict_map = dict(zip(slang_dict['slang'], slang_dict['word']))

stop_words = set(stopwords.words('english'))
stop_words.add("br")
stop_words.add("wikipedia")
stop_words.add("url")

max_length=66
app = Flask(__name__)

#Loading Binary classification model
model = keras.models.load_model('SavedBinaryAbussiveDetectionModel_V.01')

#Loading tockenizer for multilable classifier.
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer_multilabel = pickle.load(handle)
    
#loading Multi-Label classification model

def Custom_Hamming_Loss(y_true, y_pred):
  tmp = K.abs(y_true-y_pred)
  return K.mean(K.cast(K.greater(tmp,0.5),dtype=float))

#MultiLablelClassification_model = keras.models.load_model('SavedMultiLabelAbussiveDetectionModel_V.01')

MultiLablelClassification_model = keras.models.load_model('SavedMultiLabelAbussiveDetectionModel_V.01',custom_objects={ 'Custom_Hamming_Loss': Custom_Hamming_Loss })

# Declaring the Prediction threshold
threshold_dict  = {}

with open("config.yml","r") as ymlfile:
    cfg = yaml.load(ymlfile)

    
scenarioParms = cfg[cfg['type']['category']]

threshold_dict['toxic'] = scenarioParms['toxic']
threshold_dict['severe_toxic'] = scenarioParms['severe_toxic']
threshold_dict['obscene'] = scenarioParms['obscene']
threshold_dict['threat'] = scenarioParms['threat']
threshold_dict['insult'] = scenarioParms['insult']
threshold_dict['identity_hate'] = scenarioParms['identity_hate']



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/Index.html')
def backHome():
    return render_template('index.html')

@app.route('/MultilabelClassification.html')
def MultilabelClassification():
    return render_template('MultilabelClassification.html')
    
@app.route('/Twitter_scrapping.html')
def TwitterDataScrapping():
    return render_template('Twitter_scrapping.html')
    
@app.route('/AboutUs.html')
def aboutUs():
    return render_template('AboutUs.html')

# this function will clean the text
def text_cleaning(text):
    retweet_user = ['rt', 'user']
    if text:
        text = ' '.join(text.split('.'))
        text = re.sub('\/', ' ', text)
        text = re.sub(r'\\', ' ', text)
        text = re.sub(r'((www\.[^\s]+)|(http)\S+)', '',text)
        text = normalize_slang(text)
        for word in retweet_user:
            text = re.sub(word,'', text)
            text = re.sub(word.upper(),' ',text)
        #text = re.sub(r'\s+', ' ', re.sub('[^A-Za-z]', ' ', text.strip().lower())).strip()
        text = re.sub(r'\W+', ' ', text.strip().lower()).strip()
        text = re.sub(r"[0-9]", '', text)
        text = [word for word in text.split() if word not in stop_words]
        return text
    return []
    

def normalize_slang(text):
    return ' '.join([slang_dict_map[word] if word in slang_dict_map else word for word in text.split(' ')])
    
def get_word2vec_enc(tweets):
    """
    get word2vec value for each word in sentence.
    concatenate word in numpy array, so we can use it as RNN input
    """
    encoded_tweets = []
    for tweet in tweets:
        tokens = tweet.split(" ")
        word2vec_embedding = embed(tokens)
        encoded_tweets.append(word2vec_embedding)
    return encoded_tweets
    
def get_padded_encoded_tweets(encoded_tweets):
    """
    for short sentences, we prepend zero padding so all input to RNN has same length
    """
    padded_tweets_encoding = []
    for enc_tweet in encoded_tweets:
        zero_padding_cnt = max_length - enc_tweet.shape[0]
        pad = np.zeros((1, 250))
        for i in range(zero_padding_cnt):
            enc_tweet = np.concatenate((pad, enc_tweet), axis=0)
        padded_tweets_encoding.append(enc_tweet)
    return padded_tweets_encoding

#multilabel class labelling based on the prediction
def get_toxictype(i,act_or_pred):
    
    l=[]
    count=0
    
    if sum(act_or_pred[:10][i])==0:
        l.append('safe comment')
        
    else:
        for j in range(len(act_or_pred[:10][i])):
            if act_or_pred[:10][i][j]==1 and count == 0:
                l.append('toxic')
            elif act_or_pred[:10][i][j]==1 and count == 1:
                l.append('severe_toxic')
            elif act_or_pred[:10][i][j]==1 and count == 2:
                l.append('obscene')
            elif act_or_pred[:10][i][j]==1 and count == 3:
                l.append('threat')
            elif act_or_pred[:10][i][j]==1 and count == 4:
                l.append('insult')
            elif act_or_pred[:10][i][j]==1 and count == 5:
                l.append('identity_hate')
            
            count=count+1
            
    return l
    
#Predicting binary classification label on Twitter DataFrame

def get_toxictype(act_or_pred):
    act_or_pred = act_or_pred[0]
    dict_val = list(threshold_dict.values())
    dict_key = list(threshold_dict.keys())
    
    l = []
    for i,value in enumerate(act_or_pred):
            if act_or_pred[i] >= dict_val[i]:
                l.append(dict_key[i])
    if not l:
        l.append('Safe comment')      
    return l
    


def predict_twitterData(df):

    df=df[['username','text']]
    
    
    
    df['tweets'] = df['text'].apply(lambda x: ' '.join(text_cleaning(x)))
    
    
    # encode words into word2vec
    tweets = df['tweets'].tolist()
    
    encoded_tweets = get_word2vec_enc(tweets)
    #applying padding for the shorter tweets
    #max_length = get_max_length(df['tweets'])
    padded_encoded_tweets = np.array(get_padded_encoded_tweets(encoded_tweets))
   
    y_hat=model.predict(padded_encoded_tweets)
   
    y_prediction=[]
    nos_offensiveTweets=0
    for item in y_hat:
        if item.argmax()==0:
            y_prediction.append('Offensive content present in the tweet')
            nos_offensiveTweets=nos_offensiveTweets+1
        else:
            y_prediction.append('Safe Tweet')
    y_prediction=pd.DataFrame(y_prediction,columns=['prediction'])
       
    predict_label = np.hstack((df.username[:, np.newaxis],df.text[:, np.newaxis],df.tweets[:, np.newaxis], y_prediction))
    subm = pd.DataFrame(predict_label, columns = ['username','text','Tweet', 'Precense of Abusive content'])
    subm.to_csv('tiwtter_scrapping_binary_prediction.csv', index = False)
    #x=subm['Presense of Abusive content'].value_counts()
    total_tweets=len(tweets)
   
    return total_tweets,nos_offensiveTweets

@app.route('/predict',methods=['POST'])
#Predicting binary classification label on given tweet.
def predict():
    '''
    For rendering results on HTML GUI
    
    '''
    
    tweets =  request.form['Tweet']
    EneteredTweet=tweets
    render_template('index.html', entered_text=tweets)
    tweets=' '.join(text_cleaning(tweets))
    tweets=[tweets] 
    vector=get_word2vec_enc(tweets)
    vector_padded=np.array(get_padded_encoded_tweets(vector))
    y_hat=model.predict(vector_padded)
    y_prediction=[]
    for item in y_hat:
        if item.argmax()==0:
            y_prediction.append('Offensive content present in the tweet')
        else:
            y_prediction.append('No presenence of offensive content')

    return render_template('index.html', prediction_text=' {}'.format(y_prediction[0]),EneteredTweet=' {}'.format(EneteredTweet))


@app.route('/multilabelclassification',methods=['POST'])
#Predicting multilabel classification on the given tweet
def multilabelclassification():
    '''
    For rendering results on HTML GUI
    
    '''
    
    tweets =  request.form['multiTweet']
    EneteredTweet=tweets
    render_template('MultilabelClassification.html', entered_text=tweets)
    tweets=' '.join(text_cleaning(tweets))
    tweets=[tweets] 
    comment_sequence = tokenizer_multilabel.texts_to_sequences(pd.Series(tweets))
    comment_padded = pad_sequences(comment_sequence, maxlen = 1000,
                                padding = 'pre',
                                truncating='pre')
    predicted_comment = MultiLablelClassification_model.predict(comment_padded)
    commentlabels = (predicted_comment > 0.5).astype(np.int)
    
    return render_template('MultilabelClassification.html', multiLabelPrediction_text=' {}'.format(get_toxictype(predicted_comment)),EneteredTweet=' {}'.format(EneteredTweet))

# function to perform data extraction 
@app.route('/twitterdatascrapping',methods=['POST'])
def twitterdatascrapping():
    
    hashtag =  request.form['tweetHashtag']
    date_since= request.form['dateSince']
    numtweet= int(request.form['numberTweets'])
    scrape(hashtag, date_since, numtweet)
    twitter_data=pd.read_csv('scraped_tweets.csv')
    
    total_tweets,offensive_cnt=predict_twitterData(twitter_data)
    twitter_data_prediction=pd.read_csv('tiwtter_scrapping_binary_prediction.csv')
    offensive_tweets=twitter_data_prediction[twitter_data_prediction['Precense of Abusive content']=='Offensive content present in the tweet']
    offensive_tweets=pd.DataFrame(offensive_tweets[['username','text']])
    offensive_tweets.reset_index(drop=True, inplace=True)
    #twitter_data['tweets'] = twitter_data['text'].apply(clean_text)
    
    return render_template('Twitter_scrapping.html', Scrapping_Status=' {}'.format('Scraping completed'),hashtag_tweets=' {}'.format(hashtag),nos_tweets=' {}'.format(total_tweets),nos_Offensive_tweets=' {}'.format(offensive_cnt),tables=[(offensive_tweets.to_html(classes='data',justify="left")).replace('<tr>', '<tr align="left">')], titles=offensive_tweets.columns.values)

def scrape(words, date_since, numtweet):
    consumer_key = "tfh4Ej9T3wcJpNWl8x6wPOkOb"
    consumer_secret = "oVIQtmW9GPlhptYqx7mcgyjb6iJcIgXdiX1OoBkbhV0w4P0g1y"
    access_key = "1274716314377084928-kDdmBZ2cPfvFm8sBCG4tTfKw3TL8o9"
    access_secret = "lI2tB9RkWaLgwD47cV58ZJBlBSbz1aQhlCPLPctppVHPk"
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret) 
    auth.set_access_token(access_key, access_secret) 
    api = tweepy.API(auth) 
      
    # Creating DataFrame using pandas 
    db = pd.DataFrame(columns=['username', 'description', 'location', 'following', 
                               'followers', 'totaltweets', 'retweetcount', 'text', 'hashtags']) 
      
    # We are using .Cursor() to search through twitter for the required tweets. 
    # The number of tweets can be restricted using .items(number of tweets) 
    tweets = tweepy.Cursor(api.search, q=words, lang="en", 
                           since=date_since, tweet_mode='extended').items(numtweet) 
     
    # .Cursor() returns an iterable object. Each item in  
    # the iterator has various attributes that you can access to  
    # get information about each tweet 
    list_tweets = [tweet for tweet in tweets] 
      
    # Counter to maintain Tweet Count 
    i = 1  
      
    # we will iterate over each tweet in the list for extracting information about each tweet 
    for tweet in list_tweets: 
        username = tweet.user.screen_name 
        description = tweet.user.description 
        location = tweet.user.location 
        following = tweet.user.friends_count 
        followers = tweet.user.followers_count 
        totaltweets = tweet.user.statuses_count 
        retweetcount = tweet.retweet_count 
        hashtags = tweet.entities['hashtags'] 
          
        # Retweets can be distinguished by a retweeted_status attribute, 
        # in case it is an invalid reference, except block will be executed 
        try: 
            text = tweet.retweeted_status.full_text 
        except AttributeError: 
            text = tweet.full_text 
        hashtext = list() 
        for j in range(0, len(hashtags)): 
            hashtext.append(hashtags[j]['text']) 
          
        # Here we are appending all the extracted information in the DataFrame 
        ith_tweet = [username, description, location, following, 
                     followers, totaltweets, retweetcount, text, hashtext] 
        db.loc[len(db)] = ith_tweet 
          
        # Function call to print tweet data on screen 
        #printtweetdata(i, ith_tweet) 
        i = i+1
    filename = 'scraped_tweets.csv'
      
    # we will save our database as a CSV file. 
    db.to_csv(filename) 



if __name__ == "__main__":
    app.run(debug=True)
    
    