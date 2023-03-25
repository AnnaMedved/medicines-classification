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

from nltk.stem.snowball import SnowballStemmer 
from tqdm.auto import tqdm
from nltk.stem import *
from nltk.corpus import stopwords

from transformers import AutoTokenizer


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

def residual_preprocess(df, col): 
    residual = []

    for text in df[col]: 
        residual.append(
            ' '.join([w for w in text.split() if len(w) not in (2, 3)])
        )
    return residual 

def word_tokenizing(df: pd.DataFrame, feature_col: str, labels_col: str): 
    def tokenize_function(text): 
        return tokenizer(text, padding="max_length", truncation=True)

    def prepare_label(label, arr=np.zeros(len(df[labels_col].unique()))): 
        """Transform label as label position"""
        return np.array([1 if num==label else 0 for num, el in enumerate(arr)])
    
    labels = np.array(df[labels_col] - 1)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    encoding_series = df[feature_col].map(tokenize_function)

    for num, encoded in enumerate(encoding_series): 
        encoded['labels'] = labels[num]

    encoding_df = pd.DataFrame(
        data=[v.values() for v in encoding_series], 
        columns=encoding_series[0].keys()
    )
    
    encoding_df['labels'] = encoding_df['labels'].map(prepare_label)

    return tokenizer, encoding_df

def all_preprocessing(data_name: str, feature_columns: list, 
                      new_data_name: str):

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
             '-', '№', '—'
             ]
            )
        
        # Cyrillic and latin single characters: 
        a = ord('а')
        cyrillic = [chr(i) for i in range(a,a+32)]
        latin = list(string.ascii_lowercase)

        russian_stopwords.extend(cyrillic)
        russian_stopwords.extend(latin)

        df[f'deleted_sw{num}'] = stop_words_deleting(
            df=df, 
            prep_col=f'prep_text_{num}', 
            stopwords=russian_stopwords
            )
        
        df[f'deleted_sw{num}'] = residual_preprocess(
            df=df, col=f'deleted_sw{num}'
            )
        
        # Tokenizing: 
        tokenizer, encoded_dataset = word_tokenizing(
            df=df, feature_col=f'deleted_sw{num}', 
            labels_col='Подуровень'
            )
        
        features.append(f'text_tok_{num}')
        # features.append(f'lemm_{num}')

    # ============================== CHECKPOINT =============================
    # if not os.path.exists(new_data_name): 
    checkpoint_file(df=encoded_dataset, file_name=new_data_name)
    # =======================================================================
    
    return encoded_dataset, features, tokenizer



    