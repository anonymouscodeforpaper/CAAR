#!/usr/bin/env python
# coding: utf-8






from train import train_pre
import argparse
from pre_traitement import read_data




parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='Frappe', help='selection of dataset')
parser.add_argument('--n_epochs', type=int, default=100, help='the number of epochs')
parser.add_argument('--dim', type=int, default=128, help='dimension of user and entity embeddings')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--l2_weight', type=float, default=5e-6, help='weight of l2 regularization')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--context_or', type=str, default=True, help = 'Contextualized or not')






args = parser.parse_args()

read_data(args) ### This is to generate the processed data set for learning CAAR





train_pre(args,verbos = True)






