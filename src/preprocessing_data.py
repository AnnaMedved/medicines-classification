import os 
import re
import string
import pandas as pd 
import pandas as pd
import numpy as np
import string
import nltk

from nltk.corpus import stopwords
nltk.download("stopwords")
nltk.download('punkt')

# from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from nltk.stem import *
from nltk.corpus import stopwords

# from transformers import AutoTokenizer
from pymystem3 import Mystem


def remove_stop_words(text, stopwords_list):
    wds = [w for w in text.split() if w not in stopwords_list and w != ' ']

    return " ".join(wds)

def checkpoint_file(df, file_name): 
    return df.to_csv(file_name)

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

def stop_words_deleting(df: pd.DataFrame, prep_col: str): 
    """Deleting a whole stopwords with abbreviation checking"""

    # def delete_stop_words(text: str, stopwords): 
    #     text =' '.join(
    #             [word.lower() for word in text.split() 
    #             if (word.lower() not in stopwords) & 
    #             (word not in string.punctuation)]
    #         )
    #     return text
    
    russian_stopwords = stopwords.words("russian")
    russian_stopwords.extend(
        ['•', '…', '«', '»', '...', 'т.д.', 'т', 'д', 'так', 
        'нон', 'сыр', 'сур', 'сол', 'мг', 'доз', 'бет', 
        '-', '№', '—'
        ]
    )
        
    # Cyrillic and latin single characters: 
    a = ord('а')
    cyrillic = [chr(i) for i in range(a,a+32)]
    latin = list(string.ascii_lowercase)
    russian_stopwords.extend(cyrillic)
    russian_stopwords.extend(latin)

    without_sw = df[prep_col].map(lambda text: ' '.join(
                [word.lower() for word in text.split() 
                if (word.lower() not in russian_stopwords) & 
                (word not in string.punctuation)]
            ))
    return without_sw

def residual_preprocess(df: pd.DataFrame, col: str): 
    def residual(text): 
        return ' '.join([w for w in text.split() if len(w) not in (2, 3)])
    
    return np.array(df[col].map(residual), dtype=str)

def all_preprocessing(data_name: str, feature_columns: list, 
                      new_data_name: str):

    df = loading_data(data_name)
    # df = df.loc[:, feature_columns.extend('Подуровень')]

    if len(feature_columns) == 2: 
        feature1, feature2 = feature_columns[0], feature_columns[1]
        df[feature1] = df[feature1] + df[feature2]
        feature_column = feature1
    elif len(feature_columns) == 1: 
        feature_column = feature_columns[0]
    elif isinstance(feature_columns, 'str'): 
        feature_column = feature_columns

    prep_text = [
        remove_multiple_spaces(
        remove_numbers(
        remove_punctuation(text)))
        for text in tqdm(df[feature_column])
    ]
    df[f'prep_text'] = prep_text

    # stemmer = SnowballStemmer("russian") 

    df[f'deleted_sw'] = stop_words_deleting(
        df=df, 
        prep_col=f'prep_text'
    )
        
    df[f'deleted_sw'] = residual_preprocess(
        df=df, col=f'deleted_sw'
    )

        
    # df['text_lemm'] = lemm_texts_list
    
    return df