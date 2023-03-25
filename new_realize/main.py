import os 
from src.preprocessing_data import all_preprocessing

if __name__ == '__main__': 

    data_name = 'data.xlsx'

    feature_column = ['Правило взаимодействия (обр.)', 'Исходники (обр.)']
    new_data_name = 'encoded.csv'

    df, features = all_preprocessing(
        os.path.join('data', data_name), 
        feature_column, 
        os.path.join('data', new_data_name)
        ) 

    print('Working folder is ', os.getcwd())