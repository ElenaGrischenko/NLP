#!/usr/bin/env python
# coding: utf-8

# In[145]:


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


# In[ ]:


stop_words = get_stop_words('uk')


# In[111]:


#попередня обробка тексту
def ClearText(text):
    #переведення до нижнього регістру всіх слів
    cleartext = text.lower()
    #print(cleartext)
    #прибирання пустих рядків та розрив рядків
    cleartext = re.sub('\-\s\r\n\s{1,}|\-\s\r\n|\r\n', '', cleartext) 
    #залишаємо лише слова, прибираємо пунктуацію та числа
    cleartext = re.sub('[.,:;_%©?*,!@#$%^&()\d]|[+=]|[[]|[]]|[/]|"|\s{2,}|-', ' ', cleartext) #deleting symbols  
    remove_digits = str.maketrans('', '', digits)
    #cleartext = cleartext.translate(remove_digits)
    cleartext = cleartext.replace("\\", "")
    cleartext = cleartext.rstrip()
    #прибираємо зайві пробіи
    cleartext = re.sub(" +", " ", cleartext)
    #ділимо речення на список слів, розбиваємо по пробілам
    cleartext = re.split(" ", cleartext)
    #лематизація слів
    cleartext = [simplemma.lemmatize(word, lang='uk') for word in cleartext]
    #прибираємо стопслова
    cleartext = [word for word in cleartext if word not in stop_words]
    #прибираємо слова, довжина який менше 3 букв
    cleartext = [word for word in cleartext if len(word) > 3]
    return ' '.join(cleartext)


# In[97]:


print(stop_words)


# In[74]:


df_train = pd.read_excel('C:/Users/helen/OneDrive/Рабочий стол/пары/NLP_train.xlsx')
df_test = pd.read_excel('C:/Users/helen/OneDrive/Рабочий стол/пары/NLP_test.xlsx')
df_train.head()


# In[112]:


df_train['ClearText'] = df_train.apply(lambda x: ClearText(x['Comment']), axis=1)
df_test['ClearText'] = df_test.apply(lambda x: ClearText(x['Comment']), axis=1)


# In[115]:


df_train


# In[124]:


count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(df_train['ClearText'])
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)


# # MultinomialNB

# In[128]:


text_clf = Pipeline([('vect', CountVectorizer()),
                      ('tfidf', TfidfTransformer()),
                      ('clf', MultinomialNB()),
])
text_clf = text_clf.fit(df_train['ClearText'], df_train['Category'])


# In[131]:


predicted = text_clf.predict(df_test['ClearText'])
np.mean(predicted == df_test['Category'])


# # SVM

# In[140]:


text_clf_svm = Pipeline([('vect', CountVectorizer()),
                      ('tfidf', TfidfTransformer()),
                      ('clf-svm', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3,  random_state=42)),
])
train_svm = text_clf_svm.fit(df_train['ClearText'], df_train['Category'])


# In[141]:


predicted_svm = text_clf_svm.predict(df_test['ClearText'])
np.mean(predicted_svm == df_test['Category'])


# In[143]:


predicted_svm


# # LogisticRegression

# In[146]:


text_clf_log_reg = Pipeline([('vect', CountVectorizer()),
                      ('tfidf', TfidfTransformer()),
                      ('clf-svm', LogisticRegression(verbose=1, solver='liblinear',random_state=0, C=5, penalty='l2',max_iter=1000)),
])
train_log_reg = text_clf_log_reg.fit(df_train['ClearText'], df_train['Category'])


# In[147]:


predicted_log_reg = text_clf_log_reg.predict(df_test['ClearText'])
np.mean(predicted_log_reg == df_test['Category'])


