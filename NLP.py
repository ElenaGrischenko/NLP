#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re
import pandas as pd
import numpy as np
import simplemma
from stop_words import get_stop_words
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression


# In[2]:


stop_words = get_stop_words('uk')


# In[3]:


#попередня обробка тексту
def ClearText(text):
    #переведення до нижнього регістру всіх слів
    cleartext = text.lower()
    #print(cleartext)
    #прибирання пустих рядків та розрив рядків
    cleartext = re.sub('\-\s\r\n\s{1,}|\-\s\r\n|\r\n', '', cleartext) 
    #залишаємо лише слова, прибираємо пунктуацію та числа
    cleartext = re.sub('[.,:;_%©?*,!@#$%^&()\d]|[+=]|[[]|[]]|[/]|"|\s{2,}|-', ' ', cleartext) #deleting symbols  
    #cleartext = cleartext.translate(remove_digits)
    cleartext = cleartext.replace("\\", "")
    cleartext = cleartext.rstrip()
    #прибираємо зайві пробіи
    cleartext = re.sub(" +", " ", cleartext)
    #ділимо речення на список слів, розбиваємо по пробілам
    cleartext = re.split(" ", cleartext)
    #прибираємо стопслова
    cleartext = [word for word in cleartext if word not in stop_words]
    #прибираємо слова, довжина який менше 3 букв
    cleartext = [word for word in cleartext if len(word) > 3]
    #лематизація слів
    cleartext = [simplemma.lemmatize(word, lang='uk') for word in cleartext]
    return ' '.join(cleartext)


# In[4]:


print(stop_words)


# In[5]:


df_train = pd.read_excel('C:/Users/helen/OneDrive/Рабочий стол/пары/NLP_train.xlsx')
df_test = pd.read_excel('C:/Users/helen/OneDrive/Рабочий стол/пары/NLP_test.xlsx')
df_train.head()


# In[41]:


df = pd.read_excel('C:/Users/helen/OneDrive/Рабочий стол/пары/NLP_all.xlsx')
df = df.sample(frac=1) 
df


# In[42]:


df_train = df.iloc [:round(df.shape[0]*0.75)]
df_test = df.iloc [round(df.shape[0]*0.75):]


# In[43]:


df_train['ClearText'] = df_train.apply(lambda x: ClearText(x['Comment']), axis=1)
df_test['ClearText'] = df_test.apply(lambda x: ClearText(x['Comment']), axis=1)


# In[13]:


df_test


# In[44]:


count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(df_train['ClearText'])
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)


# # MultinomialNB

# In[63]:


text_clf = Pipeline([('vect', CountVectorizer()),
                      ('tfidf', TfidfTransformer()),
                      ('clf', MultinomialNB()),
])
text_clf = text_clf.fit(df_train['ClearText'], df_train['Category'])


# In[64]:


predicted = text_clf.predict(df_test['ClearText'])
np.mean(predicted == df_test['Category'])


# # SVM

# In[39]:


text_clf_svm = Pipeline([('vect', CountVectorizer()),
                      ('tfidf', TfidfTransformer()),
                      ('clf-svm', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3,  random_state=42)),
])
train_svm = text_clf_svm.fit(df_train['ClearText'], df_train['Category'])


# In[40]:


predicted_svm = text_clf_svm.predict(df_test['ClearText'])
np.mean(predicted_svm == df_test['Category'])


# In[67]:


predicted_svm


# # LogisticRegression

# In[45]:


text_clf_log_reg = Pipeline([('vect', CountVectorizer()),
                      ('tfidf', TfidfTransformer()),
                      ('clf-svm', LogisticRegression(verbose=1, solver='liblinear',random_state=0, C=5, penalty='l2',max_iter=1000)),
])
train_log_reg = text_clf_log_reg.fit(df_train['ClearText'], df_train['Category'])


# In[46]:


predicted_log_reg = text_clf_log_reg.predict(df_test['ClearText'])
np.mean(predicted_log_reg == df_test['Category'])


# In[73]:


text_clf_log_reg1 = Pipeline([('vect', CountVectorizer()),
                      ('tfidf', TfidfTransformer()),
                      ('clf-svm', LogisticRegression(verbose=1, solver='liblinear',random_state=0, C=20, penalty='l2',  max_iter=1000)),
])
train_log_reg1 = text_clf_log_reg1.fit(df_train['ClearText'], df_train['Category'])
predicted_log_reg1 = text_clf_log_reg1.predict(df_test['ClearText'])
np.mean(predicted_log_reg1 == df_test['Category'])


# In[47]:


predicted_log_reg


# In[48]:


predicted_log_reg_df = pd.DataFrame(predicted_log_reg, columns=['predictions'])
df_test.index = range(0, df_test.shape[0])
result = pd.concat([predicted_log_reg_df, df_test], axis=1)
result.to_csv('C:/Users/helen/OneDrive/Рабочий стол/пары/prediction.csv')
result


# # Metrics

# In[49]:


pred_labels = np.asarray(list(result['predictions']))
true_labels = np.asarray(list(result['Category']))


# In[50]:


# True Positive (TP): we predict a label of 1 (positive), and the true label is 1.
TP = np.sum(np.logical_and(pred_labels == 'Toxic', true_labels == 'Toxic'))
 
# True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
TN = np.sum(np.logical_and(pred_labels == 'Not_Toxic', true_labels == 'Not_Toxic'))
 
# False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
FP = np.sum(np.logical_and(pred_labels == 'Toxic', true_labels == 'Not_Toxic'))
 
# False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
FN = np.sum(np.logical_and(pred_labels == 'Not_Toxic', true_labels == 'Toxic'))


# In[51]:


print('TP = {}, TN = {}, FP = {}, FN = {}'.format(TP, TN, FP, FN))


# In[26]:


list(true_labels).count('Toxic')


# In[27]:


list(true_labels).count('Not_Toxic')


# In[52]:


Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1_Score = 2 * (Precision * Recall) / (Precision + Recall)
F1_Score


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




