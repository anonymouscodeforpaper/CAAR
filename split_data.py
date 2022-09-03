# -*- coding: utf-8 -*-



import pandas as pd
import numpy as np
import sklearn


def read_knowledge(args): ### Read attributes of items
    if args.name == 'Frappe':
        meta_app_knowledge = pd.read_csv('data/meta_app.csv')
        n_entities = max(meta_app_knowledge['rating']) + 1
        n_rel = meta_app_knowledge.shape[1] - 1
    if args.name == 'Yelp':
        meta_app_knowledge = pd.read_csv('data/meta_yelp.csv')
        n_entities = max(meta_app_knowledge['review_count']) + 1
        n_rel = meta_app_knowledge.shape[1] - 1
        
    return meta_app_knowledge, n_entities, n_rel

def read_context(args):  ### Read the user-item interactions and split them into training, test and validation set
    if args.name == 'Frappe':
        df = pd.read_csv('data/final_app.csv')
        n_cf = df.shape[1] - 3 ## Here, we compute the number of contextual factors
        n_users = len(df['user'].value_counts()) ## Here, we compute the number of users
        n_items = len(df['item'].value_counts()) ## Here, we compute the number of items
        n_contexts = max(df['city']) + 1 ## Here, we compute the number of contextual conditions
        df['cnt'] = (df['cnt'] - 1) / 2 - 1
        
        df_train = df.sample(frac=0.80,random_state=0,axis=0) ## 80% of the dataset is randomly splitted as training set
        df_rest = df[~df.index.isin(df_train.index)] 
        df_test = df_rest.sample(frac=0.50,random_state=0,axis=0) ## 10% of the dataset is randomly splitted as test set
        df_validation = df_rest[~df_rest.index.isin(df_test.index)]## 10% of the dataset is randomly splitted as validation set
        
        
        
        ### From line 47 to line 52, we give a new index to the dataframes
        df_validation=sklearn.utils.shuffle(df_validation)
        df_validation.index = range(len(df_validation))
        df_test=sklearn.utils.shuffle(df_test)
        df_test.index = range(len(df_test))
        df_train=sklearn.utils.shuffle(df_train)
        df_train.index = range(len(df_train))
        
        
    if args.name == 'Yelp':
        df = pd.read_csv('data/final_yelp.csv')
        n_cf = df.shape[1] - 3 ## Here, we compute the number of contextual factors
        n_users = len(df['user'].value_counts()) ## Here, we compute the number of users
        n_items = len(df['item'].value_counts()) ## Here, we compute the number of items
        n_contexts = max(df['Alone_Companion']) + 1 ## Here, we compute the number of contextual conditions
        df['cnt'] = (df['cnt'] - 1) / 2 - 1
        
        
        df_train = df.sample(frac=0.80,random_state=0,axis=0) ## 80% of the dataset is randomly splitted as training set
        df_rest = df[~df.index.isin(df_train.index)] 
        df_test = df_rest.sample(frac=0.50,random_state=0,axis=0) ## 10% of the dataset is randomly splitted as test set
        df_validation = df_rest[~df_rest.index.isin(df_test.index)]## 10% of the dataset is randomly splitted as validation set
        
        
        
        ### From line 47 to line 52, we give a new index to the dataframes
        df_validation=sklearn.utils.shuffle(df_validation)
        df_validation.index = range(len(df_validation))
        df_test=sklearn.utils.shuffle(df_test)
        df_test.index = range(len(df_test))
        df_train=sklearn.utils.shuffle(df_train)
        df_train.index = range(len(df_train))
    return df_train, df_test, df_validation, n_users, n_items, n_contexts, n_cf
