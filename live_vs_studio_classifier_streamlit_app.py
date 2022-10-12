# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 11:44:34 2022

@author: vmurc
"""

#YouTube API
from googleapiclient.discovery import build
from urllib.parse import urlparse, parse_qs

#Modules for downloading stuff off Project Gutenberg
import os, requests, glob

#Stuff for general sentiment analysis
import nltk
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger') #Used to determine context of word in sentence
nltk.download('punkt') #Pretrained model to tokenize words
nltk.download('omw-1.4')

import pandas as pd
import numpy as np

#Stuff to translate and detect languages
from deep_translator import GoogleTranslator
from langdetect import DetectorFactory,detect

#Stuff for dealing with strings and regular expressions
import re, string, contractions

#To read pickle file that has trained classifier
import pickle

#Streamlit
import streamlit as st

#These are the functions that will download the YouTube comments and place them into a .csv file
def build_service(api_key):
    '''
    To build the YT API service
    '''    
    key = api_key
    YOUTUBE_API_SERVICE_NAME = "youtube"    
    YOUTUBE_API_VERSION = "v3"    
    return build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey = key)
    
def get_id(url):
    '''
    To get the video id from the video url, example: 
    'https://www.youtube.com/watch?v=wfAPXlFu8', videoId = wfAPXlFu8
    '''   
    u_pars = urlparse(url)    
    quer_v = parse_qs(u_pars.query).get('v')    
    if quer_v:        
        return quer_v[0]    
    pth = u_pars.path.split('/')    
    if pth:        
        return pth[-1]
    
def save_to_csv(output_dict, filename):
    '''
    To save the comments + other columns to the csv file specified with name
    ''' 
    filename = filename.replace('/',"").replace('|',"").replace('"',"").replace(':',"")
    print(filename)
    output_df = pd.DataFrame(output_dict, columns = output_dict.keys())    
    output_df.to_csv('yt_comments.csv')
    #return output_df

#This function will get the relevant info from the video
def comments_helper(video_ID, yt_service, order, maxResults):	
    # put comments extracted in specific lists for each column
    comments, commentsId, likesCount, authors = [], [], [], []
                                       
    #get the first response from the YT service
    response = yt_service.commentThreads().list(
                                        part="snippet",   
                                        videoId = video_ID,   
                                        textFormat="plainText",
                                        order = order, 
                                        maxResults = maxResults,
                                        ).execute()
                                        
    page = 0
    while len(comments)<3000:    
        page += 1    
        index = 0    
        # for every comment in the response received    
        for item in response['items']:        
            index += 1
            comment    = item["snippet"]["topLevelComment"]        
            author     = comment["snippet"]["authorDisplayName"]        
            text       = comment["snippet"]["textDisplay"]        
            comment_id = item['snippet']['topLevelComment']['id']        
            like_count = item['snippet']['topLevelComment']['snippet']['likeCount']  
            
            #print(comment)
            
            # append the comment to the lists        
            comments.append(text)        
            commentsId.append(comment_id)        
            likesCount.append(like_count)        
            authors.append(author)    
            # get next page of comments    
        if 'nextPageToken' in response: 
            # can also specify if number of comments intended to collect reached like: len(comments) > 1001 
            response = yt_service.commentThreads().list(
                                                part="snippet",        
                                                videoId = video_ID,        
                                                textFormat="plainText",                                                    
                                                pageToken=response['nextPageToken'],
                                                order = order, 
                                                maxResults = maxResults
                                                ).execute()    
     
        # if no response is received, break     
        else:         
            break  
            
    # response to get the title of the video
    response_title = yt_service.videos().list(part = 'snippet', id = video_ID).execute()
    # get the video title
    video_title = response_title['items'][0]['snippet']['title']
    # return the whole thing as a dict and the video title to calling function in run.py    
    return dict({'Comment' : comments, 
                 'Author' : authors, 
                 'Comment ID' : commentsId, 
                 'Like Count' : likesCount}), video_title

#Wrapper function to get youtube comments from video
def get_comments(video_url, api_key, order = 'time', maxResults = 100):
    '''
    the function to fetch comments from the helper module for ONE video
    '''    
    # build the service for YT API    
    yt_service = build_service(api_key)    
    # extract video id    
    video_ID = get_id(video_url)    
    # get the comments    
    comments_dict, title = comments_helper(video_ID, yt_service, order , maxResults)
    n_comments = len(comments_dict['Comment'])
    if n_comments == 0:
        comments_dict = {'Comment':float('NaN'),
                         'Author': float('NaN'),
                         'Comment ID':float('NaN'),
                         'Like Count':float('NaN')}
    # save the output dict to storage as a csv file
    if(os.path.isfile(f'data/{title}.csv')):
        title = title + '_2'
        save_to_csv(comments_dict, title)
        #print('FOUND FILE!')
    else:
        #print('NO FILE!')
        save_to_csv(comments_dict, title)    
    print(f'Done for {video_url}.',title,n_comments)
    

def make_main_comment_df(url,file,api_key):
    vid_df = pd.read_csv(file)

    vid_id = get_id(url)
    
    url2 = f'https://www.googleapis.com/youtube/v3/videos?part=statistics&id={vid_id}&key={api_key}'
    rmd = requests.get(url2)
    rmd = rmd.json()
    views     = rmd['items'][0]['statistics']['viewCount']
    likes     = rmd['items'][0]['statistics']['likeCount']
    comments  = rmd['items'][0]['statistics']['commentCount']
    vid_df['n_views'] = views
    vid_df['n_likes'] = likes
    vid_df['n_comments'] = comments
    
    #Drop the unnamed column since it isn't useful
    vid_df = vid_df.drop('Unnamed: 0', axis=1,errors='ignore')

    return vid_df

#Translate comments to english
def translate_comment(source_language,comment):
    if source_language == 'en':
        return comment
    else:
        return GoogleTranslator(source='auto', target='en').translate(comment)
    
#This function removes unwanted characters from a string
def preprocess(sentence):
    sentence = str(sentence)
    #Define regex to remove characters that aren't in the standard English alphabet. Also preserve whitespace
    cleanr = re.compile("[^A-Za-z\s']")  
    cleantext = re.sub(cleanr, '', sentence)
    #remove URLs
    cleantext =re.sub(r'http\S+', '',cleantext)
    #remove numbers
    cleantext = re.sub('[0-9]+', '', cleantext)
    return "".join(cleantext)

def detect_comment_language(sentence):
    try:
        iso639lang = detect(sentence)
    except:
        iso639lang = np.nan
    return iso639lang

#Lemmatize
def lemmatize_sentence(tokens):
    lemmatizer = WordNetLemmatizer()
    lemmatized_sentence = []
    for word, tag in pos_tag(tokens):
        if tag.startswith('NN'):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'
        lemmatized_sentence.append(lemmatizer.lemmatize(word, pos))
    return lemmatized_sentence

stop_words = stopwords.words('english')
def remove_noise(tokens, stop_words = ()):

    cleaned_tokens = []

    for token, tag in pos_tag(tokens):
        token = token.replace('...',"").replace('"',"").replace("``","").replace("''","")
        token = token.replace('..',"").replace("'","")
        token = re.sub(r'\d+', '', token)
        
        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)
        
        if len(token) > 1 and (token == "“") or (token == "”") or (token == "’"):
            continue
        elif len(token) > 1 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
        
    return cleaned_tokens

#Get comments in input format required for model
def get_sentences_for_model(cleaned_tokens_list):
    for tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tokens)

#Classify the video as live or studio
def gnb_classify_comments(df):
    comments   = df['denoised_comments'].tolist()
    comments_for_model   = get_sentences_for_model(comments)
    dummy_dataset   = [(model_dict, "Live") for model_dict in comments_for_model]
    predictions = []
    for i in range(df.shape[0]):
        prediction = classifier.classify(dummy_dataset[i][0])
        predictions.append(prediction)
    predictions_df = pd.DataFrame(predictions)
    classification = predictions_df[0].value_counts().index.tolist()[0]
    return classification
    
def predict_live_studio(vid_df,translate=False):
    #Remove unwanted characters from comments prior to tokenizing
    vid_df['comment_processed'] = vid_df['Comment'].map(lambda s:preprocess(s)) 
    
    #Drop any nans
    vid_df = vid_df.dropna()  
    vid_df = vid_df.reset_index() 
    
    #Expand contractions
    vid_df['comment_processed'] = vid_df['comment_processed'].apply(lambda x: [contractions.fix(word) for word in x.split()])
    
    #Join the results from above with a space
    vid_df['comment_processed'] = [' '.join(map(str, l)) for l in vid_df['comment_processed']]
    
    if translate == True:
        #Get iso639 language code
        DetectorFactory.seed = 0
        vid_df['lang_iso639'] = vid_df['Comment'].apply(lambda x: detect_comment_language(x))
    
        #Translate comment to english
        vid_df['t_comment'] = vid_df.apply(lambda x: translate_comment(x['lang_iso639'],x['comment_processed']),axis=1)
    
        #Tokenize the cleaned comment
        vid_df['token_comments'] = vid_df['t_comment'].apply(word_tokenize)
    else:
        #Tokenize the cleaned comment
        vid_df['token_comments'] = vid_df['comment_processed'].apply(word_tokenize)
    
    #Add tags
    vid_df['tags_comments'] = vid_df['token_comments'].apply(pos_tag)   
    
    #Lemmatize
    vid_df['lemm_comments'] = vid_df['token_comments'].apply(lemmatize_sentence)
    
    #Final Denoising                                                                   
    vid_df['denoised_comments'] = vid_df['lemm_comments'].apply(remove_noise, stop_words = stop_words)       
    
    #Predict whether video is live or studio!
    prediction = gnb_classify_comments(vid_df)                                                     
    return prediction
    
#Loading up the Regression model we created
f = open('live_studio_classifier_v1.pickle', 'rb')
classifier = pickle.load(f)
f.close()

# Adding an appropriate title for the test website
st.title("Predicting if Music Video is From Live or Studio Performance")

st.markdown("This app uses a Gaussian Naive Bayes classifier to predict whether a video is from a live or studio performance. Give it a try!")

st.header('Enter a YouTube URL in the textbox below (Make sure that your video has comments :])')
#Making dropdown select box containing scale, key, and octave choices
url = st.text_input('YouTube URL:')

translate = st.checkbox('Translate comments to English? WARNING: Selecting this option increases processing time!', value = False)
col1, col2,col3 = st.columns(3)

with col1:
    pass
with col2:
    center_button = st.button('Predict Performance Type')
with col3:
    pass
    
if center_button:
    api_key = "AIzaSyAl7yjWd1qLxdmANEFNDAC828D2jFjucqY"
    #Fetch the comments from the provided YouTube URL
    get_comments(url, api_key, order = 'time', maxResults = 100)
    #Fetch the .csv file containing the YouTube comments
    file = glob.glob("yt_comments.csv")[0]
    #Make the dataframe containing comments to process
    vid_df = make_main_comment_df(url,file,api_key)
    prediction = predict_live_studio(vid_df,translate=translate)
    st.success(f'The video is from a {prediction} recording')
    st.video(url, format="video/mp4", start_time=0)
    
else:
    st.write("Waiting for URL to be entered...")