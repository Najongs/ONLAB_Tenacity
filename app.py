import gradio as gr
from transformers import BertForSequenceClassification, BertTokenizerFast, AutoModelForSequenceClassification
import torch
import numpy as np

MODEL_NAME ='Najongs/ability_classification_kr_bert'
MODEL_NAME2 ='Najongs/ability_sentiment_kr_bert'

# device = 'cuda' if torch.cuda.is_available() else 'cpu'

model1 = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME) # .to(device)
tokenizer1 = BertTokenizerFast.from_pretrained(MODEL_NAME)

model2 = BertForSequenceClassification.from_pretrained(MODEL_NAME2) # .to(device)
tokenizer2 = BertTokenizerFast.from_pretrained(MODEL_NAME2)

def greet(input_text):
    prediction_list1 = []
    prediction_list2 = []

    tokenized_sentence1 = tokenizer1(input_text, truncation=True, padding=False, max_length=128, return_tensors='pt')
    outputs1 = model1(**tokenized_sentence1)
    logits = outputs1.logits
    prediction_list1.extend(logits.tolist())
    np_list1 = np.array(prediction_list1)
    norm_np_list1 = np_list1/np.linalg.norm(np_list1)

    tokenized_sentence2 = tokenizer2(input_text, truncation=True, padding=False, max_length=128, return_tensors='pt')
    outputs2 = model2(**tokenized_sentence2)
    predictions = outputs2.logits.squeeze(dim=-1).cpu()
    prediction_list2.extend(predictions.tolist())
    output_tt = np.round(norm_np_list1*prediction_list2[0]*100, 2).tolist()
    output_text = str((output_tt[0]))

    return output_text


iface = gr.Interface(fn=greet, inputs="text", outputs="text")

iface.launch()