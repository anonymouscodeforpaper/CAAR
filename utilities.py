# -*- coding: utf-8 -*-


from sklearn.preprocessing import LabelEncoder
import torch
loss_func = torch.nn.MSELoss()
from sklearn.metrics import f1_score, roc_auc_score,accuracy_score
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

  
  
def RMSE(data, model,meta_app_knowledge): ## This function aims to compute RMSE and MAE
    model.eval()
    contexts_index = data.iloc[:, 3:].values
    contexts_index = torch.tensor(contexts_index).to(DEVICE)
    item = data['item'].values
    entities_index = meta_app_knowledge.loc[item].iloc[:, 1:].values
    entities_index = torch.tensor(entities_index).to(DEVICE)

    rating = torch.FloatTensor(data['cnt'].values).to(DEVICE)
    user = torch.LongTensor(data['user'].values).to(DEVICE)
    item = torch.LongTensor(data['item'].values).to(DEVICE)

    # Predict and calculate loss
    prediction = model(user, item, contexts_index, entities_index)
    
    rating = rating.cpu().detach()
    prediction = prediction.cpu().detach()
    rmse = loss_func(prediction, rating)
    mae = torch.nn.L1Loss()(prediction, rating)
    return  torch.sqrt(rmse), mae


def acc_auc(data,model,meta_app_knowledge): ## This function can be used to compute accuracy, f1, recall and AUC if CAAR is used in ranking prediction
    contexts_index = data.iloc[:, 3:].values
    contexts_index = torch.tensor(contexts_index).to(DEVICE)
    item = data['item'].values
    entities_index = meta_app_knowledge.loc[item].iloc[:, 1:].values
    entities_index = torch.tensor(entities_index).to(DEVICE)

    rating = torch.FloatTensor(data['cnt'].values).to(DEVICE)
    user = torch.LongTensor(data['user'].values).to(DEVICE)
    item = torch.LongTensor(data['item'].values).to(DEVICE)
    
    scores = model(user, item,contexts_index,entities_index)
    rating = rating.cpu().detach()
    scores = scores.cpu().detach()
    auc = roc_auc_score(rating, scores)
    scores[scores > 0.5] = 1
    scores[scores <= 0.5] = 0
    accuracy = accuracy_score(rating,scores)
    f1 = f1_score(y_true=rating, y_pred=scores)
    return accuracy, f1, auc
    
