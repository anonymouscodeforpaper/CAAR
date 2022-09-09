#!/usr/bin/env python
# coding: utf-8




import pandas as pd
import numpy as np
import torch
from torch import nn
import time
from model import CAAR
from utilities import RMSE
from split_data import read_knowledge,read_context





def train_pre(args, verbos=False): ## initialize and train the model
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    meta_app_knowledge, n_entities, n_rel = read_knowledge(args)
    df_train, df_test, df_validation, n_users, n_items, n_contexts, n_cf = read_context(args) ### Here, we get the training, test, and the validation set 
    
    model_val = CAAR(n_users, n_items, n_contexts, n_cf, n_entities, n_rel,args.dim,args.context_or).to(DEVICE)  ## We initialize the model
    
    
    
    loss_func = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(params=model_val.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5, threshold=0.05, threshold_mode='abs')
    
    
    
    
    
    for epoch in range(args.n_epochs):
        t1 = time.time()
        num_example = len(df_train)
        indices = list(range(num_example))
        for i in range(0, num_example, args.batch_size):
            #l2_reg = 0.0
            indexs = indices[i:min(i+args.batch_size, num_example)]
            contexts_index = df_train.iloc[:, 3:].loc[indexs].values
            contexts_index = torch.tensor(contexts_index).to(DEVICE)
            item = df_train['item'].loc[indexs].values
            entities = meta_app_knowledge.loc[item].iloc[:, 1:].values
            entities_index = torch.tensor(entities).to(DEVICE)
            item = torch.LongTensor(item).to(DEVICE)
            rating = torch.FloatTensor(
                df_train['cnt'].loc[indexs].values).to(DEVICE)
            
            user = df_train['user'].loc[indexs].values
            user = torch.LongTensor(user).to(DEVICE)
            prediction = model_val(user, item, contexts_index, entities_index)
            optimizer.zero_grad()
            
            err = loss_func(prediction, rating)
            reg_user = (model_val.user_factors(user)*model_val.user_factors(user)).sum()
            reg_item = (model_val.item_factors(item)*model_val.item_factors(item)).sum()
            
            
            if args.context_or == True: ## This is to compute the regulazation of parameters 
                reg_relation_c = (model_val.relation_c.weight * model_val.relation_c.weight).sum()
                reg_relation_k = (model_val.relation_k.weight * model_val.relation_k.weight).sum()
                reg_context = (model_val.context_factors(contexts_index) * model_val.context_factors(contexts_index)).sum()
                reg_entity = (model_val.entity_factors(entities_index) * model_val.entity_factors(entities_index)).sum()
                err = err + args.l2_weight * (reg_user + reg_item + reg_context + reg_entity + reg_relation_c + reg_relation_k)
            else:
                reg_relation_k = (model_val.relation_k.weight * model_val.relation_k.weight).sum()
                reg_entity = (model_val.entity_factors(entities_index) * model_val.entity_factors(entities_index)).sum()
                err = err + args.l2_weight * (reg_user + reg_item  + reg_entity + reg_relation_k)
                
            

            err.backward()
            optimizer.step()
        t2 = time.time()
        rmse,mae = RMSE(df_train,model_val,meta_app_knowledge)
        rmse1,mae1 = RMSE(df_test,model_val,meta_app_knowledge)
        rmse2,mae2 = RMSE(df_validation,model_val,meta_app_knowledge)
        scheduler.step(rmse1)
        
        if verbos == True:
            print("Epoch: ", epoch)
            print("Loss:",err)
            print(" RMSE in train set: ", rmse, " MAE in train set:", mae)
            print(" RMSE in valiadation set: ", rmse2, " MAE in validation set:", mae2)
            print(" RMSE in test set: ", rmse1, " MAE in test set:", mae1)
            print("Time consumed: ", t2 - t1)





