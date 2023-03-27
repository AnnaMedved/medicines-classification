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

from sklearn.model_selection import train_test_split
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

def residual_preprocess(df: pd.DataFrame, col: str): 
    def residual(text): 
        return ' '.join([w for w in text.split() if len(w) not in (2, 3)])
    
    return np.array(df[col].map(residual), dtype=str)

# def word_tokenizing(df: pd.DataFrame, feature_col: str, labels_col: str): 
#     # def tokenize_function(text): 
#     #     return tokenizer(text, padding="max_length", truncation=True)

#     # def prepare_label(label, arr=np.zeros(len(df[labels_col].unique()))): 
#     #     """Transform label as label position"""
#     #     return np.array([1 if num==label else 0 for num, el in enumerate(arr)])
#     def tokenization(txt): 
#         return tokenizer(txt, truncation=True, padding=True)
    
#     feature = df[feature_col] 
#     labels = df[labels_col]

#     # tokenizer = AutoTokenizer.from_pretrained(
#     #     "distilbert-base-uncased", use_fast=True
#     #     )
#     # X_train, X_test, y_train, y_test = train_test_split(
#     #     feature, labels, test_size=0.33, random_state=42
#     #     )
    
#     encodings = feature.map(tokenization)
    
    # train_encodings = tokenizer(X_train, truncation=True, padding=True)
    # val_encodings = tokenizer(X_test, truncation=True, padding=True)

    # for num, encoded in enumerate(encoding_series): 
    #     encoded['labels'] = labels[num]

    # encoding_df = pd.DataFrame(
    #     data=[v.values() for v in encoding_series], 
    #     columns=encoding_series[0].keys()
    # )
    
    # encoding_df['labels'] = encoding_df['labels'].map(prepare_label)

    # return tokenizer, encoding_df
    # return tokenizer, enc

def all_preprocessing(data_name: str, feature_columns: list, 
                      new_data_name: str):

    df = loading_data(data_name)
    # df = df.loc[:, feature_columns.extend('Подуровень')]
    feature1, feature2 = feature_columns[0], feature_columns[1]
    df[feature1] = df[feature1] + df[feature2]
    feature_column = feature1

        # Removing multiple spaces, numbers & punct: 
    prep_text = [
        remove_multiple_spaces(
        remove_numbers(
        remove_punctuation(text)))
        for text in tqdm(df[feature_column])
    ]
    df[f'prep_text'] = prep_text

    # stemmer = SnowballStemmer("russian") 
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

    df[f'deleted_sw'] = stop_words_deleting(
        df=df, 
        prep_col=f'prep_text', 
        stopwords=russian_stopwords
    )
        
    df[f'deleted_sw'] = residual_preprocess(
        df=df, col=f'deleted_sw'
    )
        
    # Tokenizing: 
    # tokenizer, train_encodings, val_encodings, y_train, y_test = word_tokenizing(
    #     df=df, feature_col=f'deleted_sw', 
    #     labels_col='Подуровень'
    # )

    # Lemmatizing: 
    mystem = Mystem() 
    lemm_texts_list = []
    for text in tqdm(df['deleted_sw']):
        try:
            text_lem = mystem.lemmatize(text)
            lemm_texts_list.append(text)
        except Exception as e:
            print(e)
        
    df['text_lemm'] = lemm_texts_list
        
    # ============================== CHECKPOINT =============================
    # if not os.path.exists(new_data_name): 
    # checkpoint_file(df=df, file_name=new_data_name)
    # =======================================================================
    
    # return tokenizer, train_encodings, val_encodings, y_train, y_test
    return df
