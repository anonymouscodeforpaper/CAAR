#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder



# In[3]:


def read_data(args): ## This function does some pretraitements to generate the data for learning CAAR
    if args.name == 'Yelp':
        df_business = read_json('yelp_academic_dataset_business.json') ## Datasets that contains the attributes of items
        df_review = read_json('yelp_academic_dataset_review.json') ## Datasest that contains user-item interactions
        review = df_review.rename(columns = {"stars":"ratings"})
        review['date'] = pd.to_datetime(review['date']) # process date
        review['day_of_week'] = review['date'].dt.dayofweek # transform to day of the week
        review['day_of_week'].value_counts() 
        review['is_weekend'] = np.where(review['day_of_week'] >=5, 'Weekend','Weekday') # transform to weekday or weekend
        review['time_of_day'] = review['date'].dt.hour # time of the day
        review['time_of_day'] = np.where(review['time_of_day'] > 21, 'Night', np.where(review['time_of_day'] > 18, 'Evening',np.where(review['time_of_day'] > 14, 'Afternoon',np.where(review['time_of_day'] > 11, 'Noon',np.where(review['time_of_day'] > 8 , 'Morning','Early_morning')))))
        df = pd.merge(review, df_business)
        
        ### From line 30 to line 47, we delete the rebundant information
        del df['review_id']
        del df['useful']
        del df['funny']
        del df['cool']
        del df['postal_code']
        del df['latitude']
        del df['longitude']
        del df['name']
        del df['address']
        del df['city']
        del df['review_count']
        del df['is_open']
        del df['attributes']
        del df['categories']
        del df['hours']
        del df['state']
        del df['stars']
        del df_usable['date']
        
        
        df_usable = df.copy()
        df_usable['Alone_Companion'] = np.where(df_usable['text'].str.contains('we|We'),'Companion','Alone') ## Here, we extracte the contextual information companion from the reviews of users by the key word "we" and "We", if they are included in the review then the user is with companion, if not, then user is alone
        df_usable = trans_frame(df_usable,3)
        df_business = dataframe_trans(df_business,1)
        
        ### From line 55 to line 56, we apply the 10-core setting
        a = df_usable['user'].value_counts()[df_usable['user'].value_counts() >= 10]
        df = df_usable[df_usable['user'].isin(list(a.index))]
        
        ## From line 59 to line 67, we give new index to users and items after the 10-core setting
        df_business = df_business[df_business['business_id'].isin(list(df['item'].values))]
        le = LabelEncoder()
        b = le.fit_transform(df['item'])
        i = df['item'].values
        dict_co = dict(zip(i, b))
        df_business['business_id'] = df_business['business_id'].map(dict_co)
        df['item'] = b
        b = le.fit_transform(df['user'])
        df['user'] = b
        
        
        
        df.to_csv('data/final_yelp.csv', index=False)
        df_business.to_csv('data/meta_yelp.csv', index=False)
        
        
        
        
        
    if args.name == 'Frappe':
        df = pd.read_csv('frappe.csv', sep="\t")
        meta_app = pd.read_csv('meta.csv', sep="\t")
        df_context = df.copy()
        del df_context['cost']
        df_context['cnt'] = df['cnt'].astype('float')
        df_context['cnt'] = df_context['cnt'].apply(np.log10) ## transformation of the number of interactions by log
        a = df_context['user'].value_counts()[df_context['user'].value_counts() >= 10] ## Here we apply the 10-core setting to ensure the quality of the dataset
        df_context = df_context[df_context['user'].isin(list(a.index))]
        
        
        meta_app = meta_app[meta_app['item'].isin(list(set(df_context['item'].values)))] ## Here we filter out the items that are excluded by the 10-core setting
        
        
        
        ### From line 95 to line 102, we give new index to each user and item after the 10-core setting
        le = LabelEncoder()
        b = le.fit_transform(meta_app['item'])
        i = meta_app['item'].values
        dict_co = dict(zip(i, b))
        df_context['item'] = df_context['item'].map(dict_co)
        meta_app['item'] = b 
        b = le.fit_transform(df_context['user'])
        df_context['user'] = b
        
        
        df_context = trans_frame(df_context,3) #we give index to all the contextual conditions (starting from 0)
        
        
        
        ### From line 32 to line 39 we process the attribute of apps namely category, number of downloads, price and ratings given by other users
        meta_app_knowledge = meta_app[['item','category','downloads','language','price','rating']]
        meta_app_knowledge['price'] = np.where(meta_app_knowledge['price'] == 'Free', 'Free',
                                       np.where(meta_app_knowledge['price'] == 'unknown','Unknown','Paid'))
        meta_app_knowledge['rating'][meta_app_knowledge['rating'] == 'unknown'] = 0
        meta_app_knowledge['rating'] = meta_app_knowledge['rating'].astype('float')
        meta_app_knowledge['rating'] = np.where(meta_app_knowledge['rating'] == 0, 'Unknown',
                                       np.where(meta_app_knowledge['rating'] >=4.3, 'High_price',
                                               np.where(meta_app_knowledge['rating'] >= 3.8, 'Mid_price','Low_price')))
        
        
        meta_app_knowledge = trans_frame(meta_app_knowledge,1)#we give index to all attributes of items (starting from 0)
        meta_app_knowledge.to_csv('data/meta_app.csv',index = False)
        df_context.to_csv('data/final_app.csv', index = False)

    



def trans_frame(df,k): ## This function aims to discretize the categorical values
    x = 0
    for col in df.columns[k:]:
        num = len(df[col].value_counts())
        le = LabelEncoder()
        y = le.fit_transform(df[col])
        df[col] = y
        df[col] = df[col] + x
        x += num
    return df

