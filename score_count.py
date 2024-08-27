import pandas as pd
import pickle
from transformers import BertForSequenceClassification, BertTokenizerFast, AutoModelForSequenceClassification
import torch
import numpy as np
from kss import split_sentences

# import gradio as gr

# DB
# 딕셔너리 구조로 표현하자면 USER ID 마다 각각 아래와 같은 변수들이 추가되어야 할 것 같다.
'''
all_data = {
    
        '00001' : { 'voteCount':0, 'sum_point':0, 'answerCount':0, 'total_work_time':0,
          'final_score' : {'passion' : 0, 'cooperation':0, 'diligence':0, 'responsibility':0, 'conductivity':0, 'leadership':0},
          'text_answser' : {'passion' : [], 'cooperation': [], 'diligence': [], 'responsibility':[], 'conductivity':[], 'leadership':[]}
          },
          
        '00002' : { 'voteCount':0, 'sum_point':0, 'answerCount':0, 'total_work_time':0,
          'final_score' : {'passion' : 0, 'cooperation':0, 'diligence':0, 'responsibility':0, 'conductivity':0, 'leadership':0},
          'text_answser' : {'passion' : [], 'cooperation': [], 'diligence': [], 'responsibility':[], 'conductivity':[], 'leadership':[]}
          },
          
        }
'''
with open('work_class.pickle', 'rb') as fr:
    work_class = pickle.load(fr)
        
# 평가가 이루어진 데이터로 현재 USER 정보를 업데이트할때 사용되는 함수이다.
def score_calculation(work_time, work_name, scored):
    
    # 직군에 따른 가중치 파일을 불러오고 이에 맞게 가중치 점수를 계산한다.
    
    work_score = {'A' : 1, 'B' : 0.97, 'C' : 0.94, 'D' : 0.91}
    
    # 받은 점수에 가중치를 곱하여 변수에 추가한다.
    weight = work_score[work_class[work_name]]*work_time
    passion = sum(scored[0:3])*weight
    cooperation = sum(scored[3:6])*weight
    diligence = sum(scored[6:9])*weight
    responsibility = sum(scored[9:12])*weight
    conductivity = sum(scored[12:15])*weight
    leadership = sum(scored[15:18])*weight
    
    # sum point는 추후에 total point를 계산하기 위해 변수로 추가 저장한다.
    sum_point = sum(scored)*weight
    
    # 최종결과 딕셔너리 파일을 OUTPUT하고 이를 데이터베이스에 USER 특성별로 더해준다.
    score_dic = {
        'passion' : passion,
        'cooperation' : cooperation,
        'diligence' : diligence, 
        'responsibility' : responsibility, 
        'conductivity' : conductivity,
        'leadership' : leadership,
        'sum_point' : sum_point
        }
    
    # 결과값을 통해서 매칭되는 특성에 더해주면 된다.
    return score_dic

MODEL_NAME ='Najongs/ability_classification_kr_bert'
model1 = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME) # .to(device)
tokenizer1 = BertTokenizerFast.from_pretrained(MODEL_NAME)

# sigmoid 구현
def stable_sigmoid(x):

    sig = np.where(x < 0, np.exp(x) / (1 + np.exp(x)), 1 / (1 + np.exp(-x)))
    return sig

# 주관식 평가를 특성별로 정리하기 위해 사용되는 함수이다.
def stable_sigmoid(x):

    sig = np.where(x < 0, np.exp(x) / (1 + np.exp(x)), 1 / (1 + np.exp(-x)))
    return sig

def greet(input_text):

    # creating a new dictionary
    personal_dict = {'근성' : 0, '열정' : 1, '협력' : 2, '책임' : 3, '창의성/생산성' : 4, '리더십' : 5}
    output_dict = {'근성' : [], '열정' : [], '협력' : [], '책임' : [], '창의성/생산성' : [], '리더십' : []}
    
    sentences = split_sentences(input_text)
    for sentence in sentences:
        prediction_list1 = []
        tokenized_sentence1 = tokenizer1(sentence, truncation=True, padding=False, max_length=128, return_tensors='pt')
        outputs1 = model1(**tokenized_sentence1)
        logits = outputs1.logits
        prediction_list1.extend(logits.tolist())
        np_list1 = np.array(prediction_list1)
        
        for i in range(0,6):
            np_list1[0][i] = stable_sigmoid(np_list1[0][i])

        # after sigmoid if values is over 0.85 getting output
        a_least = np.where(np_list1[0] > 0.85)

        # list out keys and values separately
        key_list = list(personal_dict.keys())
        val_list = list(personal_dict.values())

        # print key with values
        # 0.85를 넘는 값이 없을 경우 최댓값하나 선정 or 없는게 맞을수도
        # if len(a_least[0]) == 0:
        #     a_sort = np.argmax(np_list1[0])
        #     position = val_list.index(a_sort)
        #     output.append(key_list[position])
        
        output = []
        for j in a_least[0]:
            position = val_list.index(j)
            output.append(key_list[position])
            
        for k in output:
            output_dict[k].append(sentence) 

    return output_dict


# # 저장된 USER의 값을 호출할때 사용된다. presonal point는 원하는 특성에 따라서 DB에서 불러온다.
# def display_score(userId, personal_point):
    
#     # DB에 저장된 user ID에 각각의 특성에서 전체 근무시간을 나누어서 계산한다.
#     # 그리고 15점 만점이므로 15로 나누고 100을 곱해서 100점 만점으로 표기한다.
#     personal_score = (personal_point *100) / (total_work_time *15)
    
#     # 종합점수를 확인할때는 아래와 같은 방식을 따른다.
#     # 사실상 각각의 점수의 평균과 같다. 
#     total_score = (sum_point * 100) / (90 * voteCount)
    
#     return personal_score, total_score

# iface = gr.Interface(fn=greet, inputs="text", outputs="text")

# iface.launch()