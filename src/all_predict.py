import pickle
import pandas as pd 


def all_predictions(df: pd.DataFrame, new_data_name: str, feature_column: list):
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    predictions = model.predict(df['text_lemm'])
    pred_df = df.loc[:, feature_column]
    pred_df['Результат модели'] = predictions

    pred_df.to_excel(new_data_name)

    return pred_df