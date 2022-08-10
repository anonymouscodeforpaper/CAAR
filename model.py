# -*- coding: utf-8 -*-


import torch
from torch import nn
from torch.nn import LeakyReLU
leaky = LeakyReLU(0.1)

drop = torch.nn.Dropout(p=0.4)


class aggregator(nn.Module): ## This class computes the representation of contextual situation
    def __init__(self, input_dim, output_dim):
        super(aggregator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        #self.wei = torch.nn.Embedding(self.input_dim,self.output_dim)
        #self.bias = torch.nn.Embedding(1,self.output_dim)

    def forward(self, user, to_agg, rel):
        '''
        user: 128 * 64
        rel: 64 * 5
        to_agg: 128 * 5 * 64
        '''
        scores = torch.matmul(user, rel) # 128 * 5 See Equation 1 in the paper
        scores = leaky(scores)
        m = torch.nn.Softmax(dim=1) # 128 * 5 See Equation 2 in the paper
        scores = m(scores) # 128 * 5
        scores1 = scores.unsqueeze(1) #128 * 1 * 5
        context_agg = torch.bmm(scores1, to_agg) # 128 * 1 * 64 See Equation 3 in the paper
        return context_agg

    
class CAAR(nn.Module):
    def __init__(self, n_users, n_items, n_contexts, n_rc, n_entity, n_kc, n_factors,context_or):
        super().__init__()
        self.user_factors = torch.nn.Embedding(n_users, n_factors)
        self.item_factors = torch.nn.Embedding(n_items, n_factors)
        self.context_factors = torch.nn.Embedding(n_contexts, n_factors)
        self.relation_c = torch.nn.Embedding(n_factors, n_rc)
        self.entity_factors = torch.nn.Embedding(n_entity, n_factors)
        self.relation_k = torch.nn.Embedding(n_factors, n_kc) # 64 * 5
        self.agg = aggregator(n_factors, n_factors)
        self.context_or = context_or

    def forward(self, user, item, contexts_index, entities_index):

        entities = self.entity_factors(entities_index)  # 128 * 5 * 64
        contexts = self.context_factors(contexts_index)  # 128 * 5 * 64
        if self.context_or == True: ## if we consider users' contexts then we compute the representation of user under contextual situations using Equation 4 in the paper, note that this is simplfied implementation withour w and b
            u_nei = self.agg(self.user_factors(user), contexts,
                    self.relation_c.weight)  # 128 * 1 * 64
            u_final = u_nei + self.user_factors(user).unsqueeze(1) # 128 * 1 * 64 See Equation 4 in the paper
            u_final = leaky(u_final)
        else: ## users' contexts are not considered
            u_final = self.user_factors(user)
        importances = torch.matmul(u_final.squeeze(1), self.relation_k.weight) # 128 * 5 See Equation 5 in the paper
        importances = leaky(importances)
        m = torch.nn.Softmax(dim=1) # 128 * 5
        importances = m(importances) # 128 * 5 See Equation 6 in the paper
        scores = torch.bmm(entities,u_final.squeeze(1).unsqueeze(2)) # 128 * 5 * 1 See Equation 7 in the paper
        scores_final = torch.bmm(importances.unsqueeze(1),scores) # 128 * 1 * 1 See Equation 8 in the paper
        scores_final = scores_final.squeeze(2) # 128 * 1
        scores_final = scores_final.squeeze(1) # 128,
        return scores_final
          
    

        
      

    
    
    

    
    
    
    
