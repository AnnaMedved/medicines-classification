import os 
import string
import pandas as pd 
import unicodedata

import pandas as pd
import numpy as np
import string
import nltk
from nltk.corpus import stopwords
nltk.download("stopwords")
nltk.download('punkt')

from nltk.stem.snowball import SnowballStemmer 
from tqdm.auto import tqdm, trange
from nltk.stem import *
from nltk.corpus import stopwords
# from pymystem3 import Mystem
from nltk import word_tokenize
from pymystem3 import Mystem

import re

from itertools import chain 


def stemm_text(df, col, stemmer): 

    stemmed_texts_list = []

    for text in tqdm(df[col]):
        tokens = word_tokenize(text)    
        stemmed_tokens = [stemmer.stem(token) for token in tokens]
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

def word_tokenizing(df, prep_col): 
    sw_texts_list = []
    for text in tqdm(df[prep_col]):
        tokens = word_tokenize(text)    
        tokens = [token for token in tokens if token != ' ']
        text = " ".join(tokens)
        sw_texts_list.append(text)

    return sw_texts_list

def checkpoint_file(df, file_name): 
    return df.to_csv(file_name)

def lemmatize_text(df: pd.DataFrame, column_to_lemm: str):

    mystem = Mystem() 
    lemm_texts_list = []
    for text in tqdm(df[column_to_lemm]):
        try:
            text_lem = mystem.lemmatize(text)
            tokens = [
                token for token in text_lem if token != ' ']
            text = " ".join(tokens)
            lemm_texts_list.append(text)
        except Exception as e:
            print(e)

    return lemm_texts_list

def loading_data(data_name: str) -> pd.DataFrame: 
    """Check and load your data"""

    if data_name.endswith('.xlsx'): 
        df = pd.read_excel(data_name, engine='openpyxl')

    elif data_name.endswith('.csv'): 
        df = pd.read_csv(data_name)

    return df 

def remove_punctuation(text):
    return ''.join(
        [ch if ch not in string.punctuation else ' ' for ch in text]
        )

def remove_numbers(text):
    return ''.join([i if not i.isdigit() else ' ' for i in text])

def remove_multiple_spaces(text):
	return re.sub(r'\s+', ' ', text, flags=re.I)

def stop_words_deleting(df, prep_col, stopwords): 
    """Deleting a whole stopwords with abbreviation checking"""

    def delete_stop_words(text: str): 
        text =' '.join(
                [word.lower() for word in text.split() 
                if (word.lower() not in stopwords) & 
                (word not in string.punctuation)]
            )
        # text = remove_multiple_spaces(' '.join(
        #     [word.lower() for word in text.split() 
        #         if (not word.isupper())]))
        return text
    
    without_sw = []
    for text in df[prep_col]: 
        without_sw.append(delete_stop_words(text))

    return without_sw

def residual_preprocess(df, col): 
    residual = []
    for text in df[col]: 
        residual.append(
            ' '.join([w for w in text.split() if len(w) not in (2, 3)])
        )
    return residual 

def all_preprocessing(data_name: str, feature_columns: list):

    df = loading_data(data_name)
    features = [] # name of preprocessed columns to predict

    for num, feature_column in enumerate(feature_columns): 

        # Removing multiple spaces, numbers & punct: 
        prep_text = [
            remove_multiple_spaces(
            remove_numbers(
            remove_punctuation(text)))
            for text in tqdm(df[feature_column])
            ]
        df[f'prep_text_{num}'] = prep_text

        stemmer = SnowballStemmer("russian") 
        russian_stopwords = stopwords.words("russian")
        russian_stopwords.extend(
            ['•', '…', '«', '»', '...', 'т.д.', 'т', 'д', 'так', 
             'нон', 'сыр', 'сур', 'сол', 'мг', 'доз', 'бет', 
             '-', '№', '—', ' « ', ' » '
             ]
            )
        
        # Cyrillic and latin single characters: 
        a = ord('а')
        cyrillic = [chr(i) for i in range(a,a+32)]
        latin = list(string.ascii_lowercase)

        russian_stopwords.extend(cyrillic)
        russian_stopwords.extend(latin)

        df[f'deleted_sw_{num}'] = stop_words_deleting(
            df=df, 
            prep_col=f'prep_text_{num}', 
            stopwords=russian_stopwords
            )
        
        df[f'deleted_sw_{num}'] = residual_preprocess(
            df=df, col=f'deleted_sw_{num}'
            )

        # # Text stemming: 
        # df[f'text_stem_{num}'] = stemm_text(
        #     df=df, 
        #     col=f'deleted_sw{num}', 
        #     stemmer=stemmer)
        
        # Tokenizing: 
        # df[f'text_tok_{num}'] = word_tokenizing(
        #     df=df, prep_col=f'deleted_sw{num}'
        #     )
        
        # df[f'lemm_{num}'] = lemmatize_text(
        #     df=df,
        #     column_to_lemm=f'text_sw_{num}'
        # )
        
        features.append(f'deleted_sw_{num}')
        # features.append(f'lemm_{num}')

    # ============================== CHECKPOINT =============================
    saved = checkpoint_file(df=df, file_name='data_stemmed.csv')
    # =======================================================================
    
    return df, features 


# if __name__ == '__main__': 

    
#     # print(df.columns)
#     df_preprocessed = all_preprocessing(df, feature_column='Правило взаимодействия (обр.)') 

#     print('Working folder is ', os.getcwd())
    