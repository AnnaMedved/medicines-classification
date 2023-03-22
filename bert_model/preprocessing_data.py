import os 
import string
import pandas as pd 

import pandas as pd
import numpy as np
import string
import nltk
from nltk.corpus import stopwords
nltk.download("stopwords")

from nltk.stem.snowball import SnowballStemmer 
from tqdm.auto import tqdm, trange
from nltk.stem import *
from nltk.corpus import stopwords
# from pymystem3 import Mystem
from nltk import word_tokenize
from pymystem3 import Mystem

import re

from itertools import chain 


def remove_punctuation(text):
    return "".join(
        [ch if ch not in string.punctuation else ' ' for ch in text]
        )

def remove_numbers(text):
    return ''.join([i if not i.isdigit() else ' ' for i in text])

def remove_multiple_spaces(text):
	return re.sub(r'\s+', ' ', text, flags=re.I)

def stemm_text(df, column_to_stem, stopwords_list, stemmer): 

    stemmed_texts_list = []
    for text in tqdm(df[column_to_stem]):
        tokens = word_tokenize(text)    
        stemmed_tokens = [
            stemmer.stem(token) 
            for token in tokens if token not in stopwords_list
            ]
        text = " ".join(stemmed_tokens)
        stemmed_texts_list.append(text)
    
    return stemmed_texts_list

def remove_stop_words(text, stopwords_list):
    tokens = word_tokenize(text) 
    tokens = [
        token for token in tokens 
        if token not in stopwords_list and token != ' '
        ]

    return " ".join(tokens)

def get_removed_sw(df, prep_col, stopwords_list): 
    sw_texts_list = []
    for text in tqdm(df[prep_col]):
        tokens = word_tokenize(text)    
        tokens = [
            token for token in tokens 
            if token not in stopwords_list and token != ' '
            ]
        text = " ".join(tokens)
        sw_texts_list.append(text)

    return sw_texts_list

def checkpoint_file(df, file_name): 
    return df.to_csv(file_name)

def lemmatize_text(df: pd.DataFrame, column_to_lemm: str, stopwords):

    mystem = Mystem() 
    lemm_texts_list = []
    for text in tqdm(df[column_to_lemm]):
        try:
            text_lem = mystem.lemmatize(text)
            tokens = [
                token for token in text_lem 
                if token != ' ' and token not in stopwords
                ]
            text = " ".join(tokens)
            lemm_texts_list.append(text)
        except Exception as e:
            print(e)

    return text

def loading_data(data_name: str) -> pd.DataFrame: 
    """Check and load your data"""

    if data_name.endswith('.xlsx'): 
        df = pd.read_excel(data_name)

    elif data_name.endswith('.csv'): 
        df = pd.read_csv(data_name)

    return df 

def all_preprocessing(data_name: str, feature_columns: list):

    df = loading_data(data_name)
    features = [] # name of preprocessed columns to predict

    for num, feature_column in enumerate(feature_columns): 
        # # Removing punctuation, numbers, multiple spaces:
        # preproccessing = lambda text: (remove_multiple_spaces(
        #     remove_numbers(remove_punctuation(text))
        #     ))
        # # Adding column in initial table: 
        # df[f'preproccessed_{num}'] = list(map(
        #     preproccessing, df[feature_column]
        #     ))

        # Prep for next steps: 
        prep_text = [
            remove_multiple_spaces(
            remove_numbers(
            remove_punctuation(text.lower())))
            for text in tqdm(df[feature_column])
            ]
        df[f'prep_text_{num}'] = prep_text

        stemmer = SnowballStemmer("russian") 
        russian_stopwords = stopwords.words("russian")
        russian_stopwords.extend(['•', '…', '«', '»', '...', 'т.д.', 'т', 'д'])

        # Text stemming: 
        df[f'text_stem_{num}'] = stemm_text(
            df=df, 
            column_to_stem=f'prep_text_{num}', 
            stopwords_list=russian_stopwords,
            stemmer=stemmer
            )
        
        # Text removing stop words: 
        df[f'text_sw_{num}'] = get_removed_sw(
            df=df, prep_col=f'text_stem_{num}', 
            stopwords_list=russian_stopwords
            )
        
        df[f'lemm_{num}'] = lemmatize_text(
            df=df,
            column_to_lemm=f'text_sw_{num}', 
            stopwords=russian_stopwords
        )
        
        features.append(f'text_sw_{num}')

    # ============================== CHECKPOINT =============================
    saved = checkpoint_file(df=df, file_name='data_stemmed.csv')
    # =======================================================================
    
    # Lemmatized column only: 
    return df, features 


# if __name__ == '__main__': 

    
#     # print(df.columns)
#     df_preprocessed = all_preprocessing(df, feature_column='Правило взаимодействия (обр.)') 

#     print('Working folder is ', os.getcwd())
    