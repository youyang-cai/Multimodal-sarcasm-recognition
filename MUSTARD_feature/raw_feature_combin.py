import h5py
import numpy as np
import torch
import pickle
import pandas as pd
import openpyxl
#import _Pickle
import json

data1 = pd.read_excel('MUSTARD/MUSTARD.xlsx', header=0)
data = data1.loc[:, ['KEY','KEY1', 'SPEAKER', 'SENTENCE', 'SHOW', 'SARCASM','SENTIMENT_IMPLICIT','SENTIMENT_EXPLICIT','EMOTION_IMPLICIT','EMOTION_EXPLICIT']]
excel_keys=data['KEY']
excel_keys1=data['KEY1']
sentiment_implicit_labels=data['SENTIMENT_IMPLICIT']
sentiment_explicit_labels=data['SENTIMENT_EXPLICIT']
emotion_implicit_labels=data['EMOTION_IMPLICIT']
emotion_explicit_labels=data['EMOTION_EXPLICIT']
sarcasm_labels=data['SARCASM']
sentences=data['SENTENCE']
shows=data['SHOW']

key_list=[]
allfeature=[]
for key in excel_keys:
    if pd.notnull(key):
        if 'utterances' not in str(key):
            index = str(key).find('##')
            newkey = str(key)[0:index]
            if newkey not in key_list:
                key_list.append(newkey)
        else:
            index = str(key).find('_utterances')
            newkey = str(key)[0:index]
            if newkey not in key_list:
                key_list.append(newkey)

#make_videoIDS
print('make_videoIDS')
videoids={}
for i in range(len(key_list)):
    onekeylist=[]
    onekeylist.append(str(key_list[i]) + '_context')
    onekeylist.append(str(key_list[i]) + '_unterance')
    videoids[key_list[i]] = onekeylist
allfeature.append(videoids)


#make_speaker
print('make_speaker')
speakers={}
testkey=[]
for i in range(len(key_list)):
    onespeakerlist=[]
    onespeakerlist.append(np.array([0,1]))
    onespeakerlist.append(np.array([1,0]))
    speakers[key_list[i]]=onespeakerlist
allfeature.append(speakers)


# make_videoLabels
print('make_sarcasmsLabels')
sarcasms={}
for i in range(len(key_list)):
    onelabel=[]
    onelabel.append(0)
    for j in range(len(excel_keys)):
        if 'utterances' not in str(excel_keys[j]):
            flag=1
        else:
            index = str(excel_keys[j]).find('_utterances')
            newkey = str(excel_keys[j])[0:index]
            if newkey == key_list[i]:
                #print(excel_keys[j])
                onelabel.append(int(sarcasm_labels[j]))
    sarcasms[key_list[i]] = onelabel
allfeature.append(sarcasms)

print('make_sentiment_implicitsLabels')
sentiment_implicit={}
for i in range(len(key_list)):
    onelabel=[]
    onelabel.append(0)
    for j in range(len(excel_keys)):
        if 'utterances' in str(excel_keys[j]):
            index = str(excel_keys[j]).find('_utterances')
            newkey = str(excel_keys[j])[0:index]
            if newkey == key_list[i]:
                if int(sentiment_implicit_labels[j])==-1:
                    onelabel.append((int(0)))
                if int(sentiment_implicit_labels[j])==0:
                    onelabel.append((int(1)))
                if int(sentiment_implicit_labels[j])==1:
                    onelabel.append((int(2)))
    sentiment_implicit[key_list[i]] = onelabel
allfeature.append(sentiment_implicit)

print('make_sentiment_explicit_labels')
sentiment_explicit={}
for i in range(len(key_list)):
    onelabel=[]
    onelabel.append(0)
    for j in range(len(excel_keys)):
        if 'utterances' in str(excel_keys[j]):
            index = str(excel_keys[j]).find('_utterances')
            newkey = str(excel_keys[j])[0:index]
            if newkey == key_list[i]:
                if int(sentiment_explicit_labels[j])==-1:
                    onelabel.append((int(0)))
                if int(sentiment_explicit_labels[j])==0:
                    onelabel.append((int(1)))
                if int(sentiment_explicit_labels[j])==1:
                    onelabel.append((int(2)))
    sentiment_explicit[key_list[i]] = onelabel
allfeature.append(sentiment_explicit)

print('make_emotion_implicitsLabels')
implicits={}
for i in range(len(key_list)):
    onelabel=[]
    onelabel.append(0)
    for j in range(len(excel_keys)):
        if 'utterances' in str(excel_keys[j]):
            index = str(excel_keys[j]).find('_utterances')
            newkey = str(excel_keys[j])[0:index]
            if newkey == key_list[i]:
                onelabel.append(int(emotion_implicit_labels[j])-1)#修改
    implicits[key_list[i]] = onelabel
allfeature.append(implicits)




print('make_emotion_explicitLabels')
explicit={}
for i in range(len(key_list)):
    onelabel=[]
    onelabel.append(0)

    for j in range(len(excel_keys)):
        if 'utterances' in str(excel_keys[j]):
            index = str(excel_keys[j]).find('_utterances')
            newkey = str(excel_keys[j])[0:index]
            if newkey == key_list[i]:
                onelabel.append((int(emotion_explicit_labels[j])-1))#修改
    explicit[key_list[i]] = onelabel
allfeature.append(explicit)




print('make_Videotext')
Videotext={}
for i in range(len(key_list)):
    onevideotext=[]
    path = 'raw_feature/text_features.pkl'
    testdata = pickle.load(open(path, 'rb'), encoding='latin1')
    keys = testdata.keys()
    contexttensor=[]
    unterancetensor=[]
    for key in keys:
        if 'utterances' not in str(key):
            index = str(key).find('##')
            newkey = str(key)[0:index]
            if newkey == key_list[i]:
                contexttensor.append(testdata[key]['text'])

        else:
            index = str(key).find('_utterances')
            newkey = str(key)[0:index]
            if newkey == key_list[i]:
                unterancetensor.append(testdata[key]['text'])
    numtensor=contexttensor[0]
    for j in range(1,len(contexttensor)):
        numtensor=numtensor+contexttensor[j]
    ave_tensor=numtensor/len(contexttensor)
    Videotext[key_list[i]] = [ave_tensor,unterancetensor[0]]
allfeature.append(Videotext)


print('make audio_feature')
#make audio_feature
videoAudio={}
testkey=[]
originkey=[]

for i in range(len(excel_keys1)):
    if pd.notnull(excel_keys1[i]):
        originkey.append(excel_keys1[i])
path = 'processed_data/audio_features.pkl'
testdata = pickle.load(open(path, 'rb'), encoding='latin1')

keys = testdata.keys()
newaudio_tensor=[]
newaudio_key=[]
for i in range(len(originkey)):
    for key in keys:
        if key == originkey[i]:
            newaudio_key.append(key)
            newaudio_tensor.append(testdata[key]['librosa']) #没有visual

for i in range(len(key_list)):
    onevideoaudio = []
    contexttensor = []
    unterancetensor = []
    for j in range(len(newaudio_key)):
        if 'utterances' not in str(newaudio_key[j]):
            index = str(newaudio_key[j]).find('##')
            newkey = str(newaudio_key[j])[0:index]
            if newkey == key_list[i]:
                contexttensor.append(newaudio_tensor[j][0])  # 直接使用数组

        else:
            index = str(newaudio_key[j]).find('_utterances')
            newkey = str(newaudio_key[j])[0:index]
            if newkey == key_list[i]:
                unterancetensor.append(newaudio_tensor[j][0])  # 直接使用数组

    if contexttensor:
        numtensor = contexttensor[0]
        for j in range(1, len(contexttensor)):
            numtensor = numtensor + contexttensor[j]
        ave_tensor = numtensor / len(contexttensor)

    videoAudio[key_list[i]] = [ave_tensor, unterancetensor[0]]
allfeature.append(videoAudio)

# 以上重新运行，下面的是废物

# print('make videoVisual')
# #make videoVisual
# videoVisual={}
# path = 'processed_data/visual_features.pkl'
# f_cont = pickle.load(open(path, 'rb'), encoding='latin1')
# for i in range(len(key_list)):
#     context_tensors = []
#     utterance_tensors = []
#     for key in f_cont.keys():
#         d = f_cont[key]
#         if key == key_list[i]:
#             cont_v_tensor=np.array(d[:][0])
#             for j in range(1,len(d[:])):
#                 cont_v_tensor+=np.array(d[:][j])
#             ave_cont_v_tensor=cont_v_tensor/len(d[:])
#     for key in f_cont.keys():
#         d = f_cont[key]
#         if key == key_list[i]:
#             unter_v_tensor = np.array(d[:][0])
#             for j in range(1, len(d[:])):
#                 unter_v_tensor += np.array(d[:][j])
#             ave_unter_v_tensor = unter_v_tensor / len(d[:])
#     videoVisual[key_list[i]] = [ave_cont_v_tensor, ave_unter_v_tensor]
# allfeature.append(videoVisual)


print('make videoVisual')
# make videoVisual
videoVisual = {}
path = 'processed_data/visual_features.pkl'
f_cont = pickle.load(open(path, 'rb'), encoding='latin1')


for i in range(len(key_list)):
    # 初始化存储变量
    cont_v_tensors = []
    unter_v_tensors = []

    # 遍历字典，找到所有与 key_list[i] 相关的键
    for key, value in f_cont.items():
        if '##' in key:
            newkey = key.split('##')[0]
            if newkey == key_list[i]:
                if 'visual' in value:
                    data_tensor = value['visual'].detach().cpu().numpy() if value['visual'].is_cuda else value['visual'].numpy()
                    cont_v_tensors.append(data_tensor)
        elif '_utterances' in key:
            newkey = key.split('_utterances')[0]
            if newkey == key_list[i]:
                if 'visual' in value:
                    data_tensor = value['visual'].detach().cpu().numpy() if value['visual'].is_cuda else value['visual'].numpy()
                    unter_v_tensors.append(data_tensor)

    # 计算平均特征
    ave_cont_v_tensor = np.mean(cont_v_tensors, axis=0)
    ave_unter_v_tensor = np.mean(unter_v_tensors, axis=0)

    # 将计算的平均特征张量存储到字典中
    videoVisual[key_list[i]] = [ave_cont_v_tensor, ave_unter_v_tensor]
# if key_list:
#     first_key = key_list[0]
#     print("First key visual features dimensions:")
#     print("Context features dimension:", videoVisual[first_key][0].shape)
#     print("Utterance features dimension:", videoVisual[first_key][1].shape)


allfeature.append(videoVisual)






print('make_videoSentence')
#make_videoSentence
videoSentence={}
for i in range(len(key_list)):
    onesentence=[]
    contextsentence=''
    for j in range(len(excel_keys)):
        if 'utterances' not in str(excel_keys[j]):
            index = str(excel_keys[j]).find('##')
            newkey = str(excel_keys[j])[0:index]
            if newkey == key_list[i]:
                contextsentence+=sentences[j]
        else:
            index = str(excel_keys[j]).find('_utterances')
            newkey = str(excel_keys[j])[0:index]
            if newkey == key_list[i]:
                unterance=sentences[j]
    videoSentence[key_list[i]] = [contextsentence,unterance]
allfeature.append(videoSentence)

print('make_trainID_testID')
trainVid=set()
testVid=set()
for i in range(len(excel_keys)):
    key=excel_keys[i]
    if 'utterances' in str(excel_keys[i]) and shows[i]=='FRIENDS':
        index = str(key).find('_utterances')
        newkey = str(key)[0:index]
        testVid.add(newkey)

for i in range(len(key_list)):
    if key_list[i] not in testVid:
        trainVid.add(key_list[i])

allfeature.append(trainVid)
allfeature.append(testVid)

with open('processed_data/MUSTARD_extract_feature.pkl', 'wb') as f:
    pickle.dump(allfeature, f)
print('end...')
