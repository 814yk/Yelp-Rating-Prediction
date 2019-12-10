#!/usr/bin/env python
# coding: utf-8

# In[1]:


#pip install orjson 
#pip install tqdm
#pip install scipy
import json
import re
import numpy as np
#from tqdm import notebook
import collections
from tqdm import tqdm
from scipy import sparse


# In[2]:


#"data/stopword.list"
def get_stop_words(path):    
    stop_word = set()
    list_file = open(path, 'r').read().split("\n")
    for line in list_file:
        stop_word.add(line)
    return stop_word


# In[3]:


def tokenize(text, stop_word):
    text_tokens = []
    text = re.sub('[^\s\w]|\w*\d\w*', '', text).split()
    #reference : https://greeksharifa.github.io/%EC%A0%95%EA%B7%9C%ED%91%9C%ED%98%84%EC%8B%9D(re)/2018/08/04/regex-usage-05-intermediate/
    for token in text:
        if token not in stop_word:
            text_tokens.append(token.strip())
    return text_tokens


# In[4]:


#data_path='data/yelp_reviews_train.json'
def extract(data_path):
    tmp_token=[]
    tmp_star = []
    tmp_rating = []
    stop_word=get_stop_words("data/stopword.list")
    lines = open(data_path, 'r').read().split("\n")
    for line in tqdm(lines):
        if line == "":
            continue
        review = json.loads(line)
        str_token = tokenize(review['text'].lower(),stop_word)
        tmp_token.append(str_token)
        np_star = np.zeros(5)
        rating = int(review['stars'])
        np_star[rating - 1] = 1
        tmp_star.append(np_star)
        tmp_rating.append(rating)
    return tmp_token,tmp_star,tmp_rating


# In[5]:


#data_path='data/yelp_reviews_train.json'
def dev_extract(data_path):
    tmp_token=[]
    stop_word=get_stop_words("data/stopword.list")
    lines = open(data_path, 'r').read().split("\n")
    for line in tqdm(lines):
        if line == "":
            continue
        review = json.loads(line)
        str_token = tokenize(review['text'].lower(),stop_word)
        #print(str_token)
        tmp_token.append(str_token)
    return tmp_token


# In[6]:


token,star,rating=extract('data/yelp_reviews_train.json')


# In[7]:


'''score=[0,0,0,0,0]
for i in star:
    score+=i
print("score : ",score)
print("ratio :", score/sum(score))'''


# In[8]:


train_token=token[:int(len(token)*0.8)]
train_star=star[:int(len(token)*0.8)]
train_rating=rating[:int(len(token)*0.8)]
test_token=token[int(len(token)*0.8):]
test_star=star[int(len(token)*0.8):]
test_rating=rating[int(len(token)*0.8):]


# In[9]:


len(train_rating)


# In[10]:


def CTF_dict_new(token,CTF_vocab):
    dic={}
    C=collections.Counter(token)
    for i in set(token):
        if i in CTF_vocab:
            dic[CTF_vocab.index(i)]=C[i]
    return dic


# In[11]:


def DF_dict_new(token,DF_vocab):
    dic={}
    C=collections.Counter(token)
    for i in set(token):
        if i in DF_vocab:
            dic[DF_vocab.index(i)]=C[i]
    return dic


# In[12]:


def get_txt(path):
    tmp=[]
    f=open(path,'r')
    while True:
        line = f.readline()
        if not line: break
        tmp.append(line.rstrip('\n'))
    f.close()
    return tmp


# In[13]:


def CTF(token):
    vocab_freq={}
    for i in tqdm(range(len(token))):
        tokens=token[i]
        for w in tokens:
            try:
                vocab_freq[w]+=1
            except:
                vocab_freq[w]=1
    sorted_v = sorted(vocab_freq.items(), key=lambda kv: kv[1],reverse=True)
    vocab_freq = collections.OrderedDict(sorted_v)
    CTF_vocab=[x for x in vocab_freq]
    CTF_vocab=CTF_vocab[:2000]
    return CTF_vocab


# In[14]:


def DF(token):
    DF = {}
    for i in range(len(token)):
        tokens = token[i]
        for w in tokens:
            try:
                DF[w].add(i)
            except:
                DF[w] = {i}
    for i in DF:
        DF[i]=len(DF[i])
    sorted_df = sorted(DF.items(), key=lambda kv: kv[1],reverse=True)
    DF_freq = collections.OrderedDict(sorted_df)
    DF_vocab=[x for x in DF_freq]
    DF_vocab=DF_vocab[:2000]
    return DF_vocab


# In[15]:


def get_CTF_matrix(token,CTF_vocab):
    import time
    start=time.time()
    row_ctf=[]
    col_ctf=[]
    data_ctf=[]
    n=0
    for i in tqdm(token):
        dic=CTF_dict_new(i,CTF_vocab)
        row_ctf.extend([n]*len(dic))
        col_ctf.extend(dic.keys())
        data_ctf.extend(dic.values())
        del dic
        n+=1
    print("CTF_MATRIX DONE : ", time.time()-start)
    CTF_mtx=sparse.csr_matrix((data_ctf, (row_ctf, col_ctf)), shape=(len(token), 2000))
    return CTF_mtx


# In[16]:


def get_DF_matrix(token,DF_vocab):
    import time
    start=time.time()
    row_df=[]
    col_df=[]
    data_df=[]
    n=0
    for i in tqdm(token):
        dic=DF_dict_new(i,DF_vocab)
        row_df.extend([n]*len(dic))
        col_df.extend(dic.keys())
        data_df.extend(dic.values())
        del dic
        n+=1
    print("DF_MATRIX DONE : ",time.time()-start)
    DF_mtx=sparse.csr_matrix((data_df, (row_df, col_df)), shape=(len(token), 2000))
    return DF_mtx


# In[17]:


import random
def logistic_regression(train_mtx, list_star,rating,test_mtx,test_rating):
    label_mtx = np.array(list_star)
    # use gradient ascent to update model
    alpha = 0.003
    lamda = 0.5
    steps = 100000
    #batch_size=5000
    model_gd = gradient_ascent(train_mtx, label_mtx, alpha, lamda, steps,rating,test_mtx,test_rating)
    return model_gd


def gradient_ascent(train_mtx, label_mtx, alpha, lamda, steps,rating,test_mtx,test_rating):
    import math
    rmse_list=[]
    # initialize matrix w
    test_rating=np.array(test_rating,dtype=float)
    model_mtx = np.zeros((5, 2000))
    row_size = train_mtx.shape[0]
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
            if np.array(rmse_list).max()- np.array(rmse_list).min()<0.001:
                print('converge')
                print(step)
                print(rmse_list)
                break
        #if np.sqrt(np.sum(np.square(gradient))) < 0.00001:
            #break

    return model_mtx


# In[18]:


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
    rmse = math.sqrt(np.sum(np.square(soft_pred - eval_label)/soft_pred.shape[0]))
    return print('ACC :', acc, ' RMSE :', rmse)


# In[19]:


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


# In[20]:


def write_no_lb(preds,save_path):
    f = open(save_path, 'w')
    for line in preds:
        f.write(str(line)+" "+"0"+"\n")


# # DEV & TEST

# In[21]:


dev_token=dev_extract('data/yelp_reviews_dev.json')


# In[22]:


'''ttest_token=dev_extract('data/yelp_reviews_test.json')'''


# # DF

# In[23]:


#train_token, test_token
#train_rating test_rating
#train_star test_star


# In[24]:


DF_vocab=DF(train_token)


# In[25]:


DF_train_mtx=get_DF_matrix(train_token,DF_vocab)
DF_test_mtx=get_DF_matrix(test_token,DF_vocab)


# In[26]:


pred_df_mtx=logistic_regression(DF_train_mtx,train_star,train_rating,DF_test_mtx,test_rating)


# In[27]:


validate_model(pred_df_mtx,DF_train_mtx,train_rating)


# In[28]:


#predict(pred_df_mtx,DF_mtx,'results/train_df.txt')


# In[29]:


'''dev_DF_mtx=get_DF_matrix(dev_token,DF_vocab)'''


# In[30]:


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


# In[31]:


#test_DF_mtx=get_matrix(test_token,"DF")


# In[32]:


'''write(pred_df_mtx,dev_DF_mtx,'lr_dev_df.txt')'''


# In[33]:


#write(pred_df_mtx,test_DF_mtx,'results/lr_test_df.txt')


# In[34]:


from sklearn.svm import LinearSVC
import time
# initialise the SVM classifier
DF_classifier = LinearSVC(dual=False)

# train the classifier
start = time.time()
DF_classifier.fit(DF_train_mtx, train_rating)
print(time.time()-start)


# In[35]:


df_svm_preds = DF_classifier.predict(DF_train_mtx)
'''write_no_lb(df_svm_preds,'results/svm_train_df.txt')'''


# In[36]:


'''dev_df_svm_preds = DF_classifier.predict(dev_DF_mtx)
write_no_lb(dev_df_svm_preds,'svm_dev_df.txt')'''


# In[37]:


'''test_df_svm_preds = DF_classifier.predict(test_DF_mtx)
write_no_lb(test_df_svm_preds,'results/svm_test_df.txt')'''


# In[38]:


correct = np.sum(df_svm_preds == train_rating)
print("DF-SVM-ACC :",(correct + 0.0) / len(df_svm_preds))


# # CTF

# In[39]:


CTF_vocab=CTF(train_token)


# In[40]:


CTF_train_mtx=get_CTF_matrix(train_token,CTF_vocab)
CTF_test_mtx=get_CTF_matrix(test_token,CTF_vocab)


# In[41]:


pred_ctf_mtx=logistic_regression(CTF_train_mtx,train_star,train_rating,CTF_test_mtx,test_rating)


# In[42]:


validate_model(pred_ctf_mtx,CTF_train_mtx,train_rating)


# In[43]:


dev_CTF_mtx=get_CTF_matrix(dev_token,DF_vocab)


# In[44]:


#write(pred_ctf_mtx,dev_CTF_mtx,'lr_dev_ctf.txt')


# In[ ]:





# In[45]:


from sklearn.svm import LinearSVC
import time
# initialise the SVM classifier
CTF_classifier = LinearSVC(dual=False)

# train the classifier
start = time.time()
CTF_classifier.fit(CTF_train_mtx, train_rating)
print(time.time()-start)


# In[46]:


ctf_svm_preds = CTF_classifier.predict(CTF_train_mtx)


# In[47]:


correct = np.sum(ctf_svm_preds == train_rating)
print("DF-SVM-ACC :",(correct + 0.0) / len(ctf_svm_preds))


# In[48]:


'''dev_ctf_svm_preds = CTF_classifier.predict(dev_CTF_mtx)
write_no_lb(dev_ctf_svm_preds,'svm_dev_ctf.txt')'''


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




