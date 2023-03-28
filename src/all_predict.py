import pandas as pd 
import numpy as np
import torch 

from transformers import BertForSequenceClassification, BertTokenizer
from keras_preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from src.preprocessing_data import checkpoint_file


def all_predictions(df: pd.DataFrame, new_data_name: str, feature_column: list):

    def get_mask(sent): 
        """For each sentence create the attention mask"""
        return [int(token_id > 0) for token_id in sent]

    def convert_predictions(pred):
        """Convert numpy arrays-tensors to real predictions"""

        pred = pred[0]
        pr_max = pred.max()
        return [num for num, pr in enumerate(pred) if pr == pr_max][0] + 1 
    
    
    if len(feature_column) >= 2: 
        pred_df = df.loc[:, feature_column]
    elif len(feature_column) == 1: 
        pred_df = df.loc[:, feature_column[0]]
    else: 
        print('Некорректный ввод feature_column')

    tokenizer_name = 'DeepPavlov/rubert-base-cased-sentence'
    model_path = 'model'

    tokenizer = BertTokenizer.from_pretrained(tokenizer_name, do_lower_case=True)
    model = BertForSequenceClassification.from_pretrained(
        model_path, 
        num_labels=8,   
        local_files_only=True
        )
    
    # Create sentence and label lists
    sentences = df.deleted_sw.values

    # if 'Подуровень' in feature_column: 
    #     labels = df['Подуровень'].values
    # else: labels = [] 

    input_ids = list(map(lambda sent: tokenizer.encode(
                            sent,                      
                            add_special_tokens=True, 
                    ), sentences))

    # Pad our input tokens
    MAX_LEN = 512 
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, 
                            dtype="long", truncating="post", padding="post")


    attention_masks = list(map(lambda x: get_mask(x), input_ids))

    # Convert to tensors.
    prediction_inputs = torch.tensor(input_ids)
    prediction_masks = torch.tensor(attention_masks)
    # prediction_labels = torch.tensor(labels)

    # Set the batch size.  
    batch_size = 1

    # Create the DataLoader.
    # prediction_data = TensorDataset(prediction_inputs, prediction_masks, prediction_labels)
    prediction_data = TensorDataset(prediction_inputs, prediction_masks)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda': 
        model.cuda()
    else: model.cpu()

    model.eval()

    # Tracking variables 
    predictions = []

    # Predict 
    for batch in prediction_dataloader:
    # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask = batch
        
        # Telling the model not to compute or store gradients, saving memory and 
        # speeding up prediction
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            outputs = model(
                b_input_ids, token_type_ids=None, 
                attention_mask=b_input_mask
                )

        logits = outputs[0]

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        # label_ids = b_labels.to('cpu').numpy()
        
        # Store predictions
        predictions.append(logits)
        # true_labels.append(label_ids)

    model_preds = np.array(list(map(lambda x: convert_predictions(x), predictions)))
    # true_labels = np.array(true_labels)
    pred_df['Predictions'] = model_preds

    saved = checkpoint_file(df=pred_df, file_name=new_data_name)

    return pred_df