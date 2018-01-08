
# coding: utf-8

# In[25]:

import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns

from ipywidgets import interact
import ipywidgets as widgets
from colorama import init, Fore


# 普通にデータを読み込んで...

# In[2]:

df = pd.read_csv('../data/YouTube-Spam-Collection-v1/Youtube01-Psy.csv')


# 普通に表示すると、こうなる

# In[3]:

df


# 一個ずつ表示しようとすると...毎回引数変えないといけないので面倒

# In[4]:

def print_row(i):
    for k in df.keys():
        print('{}:\n{}'.format(k, df.loc[i,k]))
print_row(0)


# In[5]:

slidebar = widgets.IntSlider(
    value=0,
    min=0,
    max=df.shape[0]
)
interact(print_row, i=slidebar)


# In[6]:

titanic_df=pd.read_csv('../data/titanic/train.csv')


# In[8]:

col_name = list(titanic_df.describe(include=['O']).keys())

@interact(cat=['Sex', 'Embarked', 'Pclass', 'SibSp', 'Parch'])
def plot_categories(cat):
    facet = sns.FacetGrid( titanic_df)
    facet.map( sns.barplot , cat , 'Survived' )
    facet.add_legend()
    sns.plt.show()


# In[12]:

import cv2


# In[65]:

def read_img(name_i):
    img = cv2.imread(name_i)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

img = read_img('../data/Lenna.png')
plt.imshow(img)
plt.axis('off')
plt.show()


# In[69]:

@interact(th=(0,255))
def get_cont(th):
    imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(imgray,th,255,0)
    image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    img_plt=img.copy()
    img_cont = cv2.drawContours(img_plt, contours, -1, (0,255,0), 3)
    plt.imshow(img_plt)
    plt.axis('off')
    plt.show()


# In[79]:

@interact(i=slidebar)
def print_row_re(i):
    print(re.sub(r"(channel|subscribe)", Fore.RED+r'\1'+Fore.RESET, df.loc[i,"CONTENT"]))


# In[ ]:




# In[ ]:



