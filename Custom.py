#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import json
import re
import numpy as np
import collections
from tqdm import tqdm
from scipy import sparse
import operator
import random
random.seed(90)

import string
from tqdm import tqdm


# In[ ]:


#"data/stopword.list"
def get_stop_words(path):    
    stop_word = set()
    list_file = open(path, 'r').read().split("\n")
    for line in list_file:
        stop_word.add(line)
    return stop_word


# In[ ]:


stopword=get_stop_words("data/stopword.list")


# In[ ]:


def tokenize(text, stop_word):
    text_tokens = []
    text = re.sub('[^\s\w]|\w*\d\w*', '', text).split()
    #reference : https://greeksharifa.github.io/%EC%A0%95%EA%B7%9C%ED%91%9C%ED%98%84%EC%8B%9D(re)/2018/08/04/regex-usage-05-intermediate/
    for token in text:
        if token not in stop_word:
            text_tokens.append(token.strip())
    str_text = ' '.join(text_tokens)
    return str_text,text_tokens


# In[ ]:


#data_path='data/yelp_reviews_train.json'
def extract(data_path):
    tmp_token=[]
    tmp_star = []
    tmp_rating = []
    tmp_text=[]
    stop_word=get_stop_words("data/stopword.list")
    lines = open(data_path, 'r').read().split("\n")
    for line in tqdm(lines):
        if line == "":
            continue
        review = json.loads(line)
        str_texts,text_tokens = tokenize(review['text'].lower(),stop_word)
        #str_texts,text_tokens = cleaning_text(review['text'])
        tmp_token.append(text_tokens)
        tmp_text.append(str_texts)
        np_star = np.zeros(5)
        rating = int(review['stars'])
        np_star[rating - 1] = 1
        tmp_star.append(np_star)
        tmp_rating.append(rating)
    return tmp_text,tmp_token,tmp_star,tmp_rating


# In[ ]:


raw_text,raw_token, raw_star, raw_rating = extract('data/yelp_reviews_train.json')


# In[ ]:


test_star=raw_star[int(len(raw_star)*0.95):]
test_token=raw_token[int(len(raw_star)*0.95):]
test_ranking=raw_rating[int(len(raw_star)*0.95):]
test_text=raw_text[int(len(raw_star)*0.95):]

train_star=raw_star[:int(len(raw_star)*0.95)]
train_token=raw_token[:int(len(raw_star)*0.95)]
train_ranking=raw_rating[:int(len(raw_star)*0.95)]
train_text=raw_text[:int(len(raw_star)*0.95)]


# In[ ]:


def write(model_mtx, test_mtx,save_path):
    row_size = test_mtx.shape[0]
    f = open(save_path, 'w')
    label = np.array([1, 2, 3, 4, 5])
    for line in range(row_size):
        exp_wx = np.exp(model_mtx * test_mtx[line, :].transpose())
        cond_prob = exp_wx / np.sum(exp_wx)
        hard_pred = np.argmax(cond_prob) + 1
        soft_pred = np.sum(label * cond_prob.transpose())
        f.write(str(hard_pred) + " " + str(soft_pred) + "\n")


# In[ ]:


def validate_model(model_mtx, eval_mtx, eval_label):
    import math
    row_size = eval_mtx.shape[0]
    exp_wx = np.exp(model_mtx * eval_mtx.transpose())
    cond_prob = exp_wx / np.sum(exp_wx, axis=0)
    hard_pred = np.argmax(cond_prob, axis=0) + 1
    correct = np.sum(hard_pred == eval_label)
    acc = (correct + 0.0) / row_size
    label = np.array([[1], [2], [3], [4], [5]])
    soft_pred = np.sum(label * cond_prob, axis=0)
    eval_label=np.array(eval_label,dtype=float)
    rmse = math.sqrt(np.sum(np.square(np.subtract(soft_pred,eval_label))/soft_pred.shape[0]))
    return print('ACC :', acc, ' RMSE :', rmse)


# In[ ]:


#data_path='data/yelp_reviews_train.json'
def dev_extract(data_path):
    tmp_token=[]
    tmp_text=[]
    stop_word=get_stop_words("data/stopword.list")
    lines = open(data_path, 'r').read().split("\n")
    for line in tqdm(lines):
        if line == "":
            continue
        review = json.loads(line)
        str_texts,text_tokens = tokenize(review['text'].lower(),stop_word)
        #print(str_token)
        tmp_token.append(text_tokens)
        tmp_text.append(str_texts)
    return tmp_text,tmp_token


# In[ ]:


dev_text,dev_token=dev_extract('data/yelp_reviews_dev.json')
ttest_text,ttest_token=dev_extract('data/yelp_reviews_test.json')


# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
tvec = TfidfVectorizer(max_features=6000,ngram_range=(1,2))
tvec.fit(train_text)

X_train_tvec = tvec.transform(train_text)
X_test_tvec=tvec.transform(test_text)


# In[ ]:


'''from sklearn.feature_extraction.text import CountVectorizer
cvec = CountVectorizer(max_features=6000,ngram_range=(1,2))
cvec.fit(train_text)

X_train_cvec = cvec.transform(train_text)
X_test_cvec = cvec.transform(test_text)'''


# In[ ]:


'''Dev_cvec=cvec.transform(dev_text)
Test_cvec=cvec.transform(ttest_text)'''


# In[ ]:


Dev_tvec=tvec.transform(dev_text)
Test_tvec=tvec.transform(ttest_text)


# In[ ]:


'''import random
def logistic_regression(train_mtx, list_star,rating,test_mtx,test_rating):
    label_mtx = np.array(list_star)
    # use gradient ascent to update model
    alpha = 0.001
    lamda = 0.3
    steps = 10000
    #batch_size=5000
    model_gd = gradient_ascent(train_mtx, label_mtx, alpha, lamda, steps,rating,test_mtx,test_rating)
    return model_gd


def gradient_ascent(train_mtx, label_mtx, alpha, lamda, steps,rating,test_mtx,test_rating):
    import math
    # initialize matrix w
    rmse_list=[]
    model_mtx = np.zeros((5, 6000))
    row_size = train_mtx.shape[0]
    test_rating=np.array(test_rating,dtype=float)
    for step in range(0, steps):
        alpha *= 1 / (1 + alpha * lamda * step)
        pick = random.sample(range(row_size), 8000)
        sgd_mtx = train_mtx[pick, :]
        sgd_label = label_mtx[pick, :]
        e_wx = np.exp(sgd_mtx * model_mtx.transpose())
        e_sum = np.sum(e_wx, axis=1)
        e_div = (e_wx.transpose() / e_sum).transpose()
        sgd_sub = np.subtract(sgd_label, e_div)
        gradient = alpha * (sgd_sub.transpose() * sgd_mtx - lamda * model_mtx)
        model_mtx += gradient

        exp_wx = np.exp(model_mtx * test_mtx.transpose())
        cond_prob = exp_wx / np.sum(exp_wx, axis=0)
        label = np.array([[1], [2], [3], [4], [5]])
        soft_pred = np.sum(label * cond_prob, axis=0)
        rmse = math.sqrt(np.sum(np.square(soft_pred - test_rating)/soft_pred.shape[0]))
        rmse_list.append(float(rmse))
        rmse_list=rmse_list[-20:]
        #print(np.array(rmse_list))
        #print(step,rmse)
        print(rmse)
        if step > 20:
            if np.array(rmse_list).max()- np.array(rmse_list).min()<0.00001:
                print('converge')
                print('step:',step)
                print('rmse:',rmse)
                break
        #if np.sqrt(np.sum(np.square(gradient))) < 0.00001:
            #break
        if step>steps-10:
            print(rmse)
    #print(rmse)
    return model_mtx'''


# In[ ]:


#pred_tfidf_mtx=logistic_regression(X_train_tvec,train_star,train_ranking,X_test_tvec,test_ranking)


# In[ ]:


'''pred_tf_mtx=logistic_regression(X_train_cvec,train_star,train_ranking,X_test_cvec,test_ranking)'''


# In[ ]:


'''write(pred_tf_mtx,Dev_cvec,'dev-predictions.txt')'''


# In[ ]:


'''write(pred_tf_mtx,Dev_cvec,'test-predictions.txt')'''


# In[ ]:


import random
def logistic_regression_tfidf(train_mtx, list_star,rating,test_mtx,test_rating):
    label_mtx = np.array(list_star)
    # use gradient ascent to update model
    alpha = 0.1
    lamda = 0.18
    steps = 10000
    #batch_size=5000
    model_gd = gradient_ascent_tfidf(train_mtx, label_mtx, alpha, lamda, steps,rating,test_mtx,test_rating)
    return model_gd


def gradient_ascent_tfidf(train_mtx, label_mtx, alpha, lamda, steps,rating,test_mtx,test_rating):
    import math
    # initialize matrix w
    rmse_list=[]
    model_mtx = np.zeros((5, 6000))
    row_size = train_mtx.shape[0]
    test_rating=np.array(test_rating,dtype=float)
    for step in range(0, steps):
        alpha *= 1 / (1 + alpha * lamda * step)
        pick = random.sample(range(row_size), 8000)
        sgd_mtx = train_mtx[pick, :]
        sgd_label = label_mtx[pick, :]
        e_wx = np.exp(sgd_mtx * model_mtx.transpose())
        e_sum = np.sum(e_wx, axis=1)
        e_div = (e_wx.transpose() / e_sum).transpose()
        sgd_sub = np.subtract(sgd_label, e_div)
        gradient = alpha * (sgd_sub.transpose() * sgd_mtx - lamda * model_mtx)
        model_mtx += gradient

        exp_wx = np.exp(model_mtx * test_mtx.transpose())
        cond_prob = exp_wx / np.sum(exp_wx, axis=0)
        label = np.array([[1], [2], [3], [4], [5]])
        soft_pred = np.sum(label * cond_prob, axis=0)
        rmse = math.sqrt(np.sum(np.square(soft_pred - test_rating)/soft_pred.shape[0]))
        rmse_list.append(float(rmse))
        rmse_list=rmse_list[-20:]
        #print(np.array(rmse_list))
        #print(step,rmse)
        print(rmse)
        if step > 20:
            if np.array(rmse_list).max()- np.array(rmse_list).min()<0.00001:
                print('converge')
                print('step : ',step)
                print('rmse : ',rmse_list)
                break
        #if np.sqrt(np.sum(np.square(gradient))) < 0.00001:
            #break
        if step>steps-10:
            print(rmse)
    #print(rmse)
    return model_mtx


# In[ ]:


pred_tfidf_mtx=logistic_regression_tfidf(X_train_tvec,train_star,train_ranking,X_test_tvec,test_ranking)


# In[ ]:


write(pred_tfidf_mtx,Dev_tvec,'dev-predictions.txt')


# In[ ]:


write(pred_tfidf_mtx,Dev_tvec,'test-predictions.txt')


# In[ ]:


#lambda 0.3 lr 0.1

