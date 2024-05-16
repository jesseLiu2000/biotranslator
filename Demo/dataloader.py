import torch
import os
import json
import wandb
import pickle
import numpy as np
import torch.nn as nn
from open_clip.utils import one_hot
from datetime import datetime
from accelerate import Accelerator
from open_clip import create_model_and_transforms, get_tokenizer, ClipLoss, ClipMetric
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import (
    TrainingArguments,
    Trainer
)
from sklearn.metrics import (
    accuracy_score,
    hamming_loss,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.utils.class_weight import compute_class_weight

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "epoch_64.pt"

model_cfg = 'PubMedBERT_512-CNN'
### create and load model ###
model, _, _ = create_model_and_transforms(
        model_cfg,
        '',
        precision='amp',
        device=torch.device('cuda'),
        jit=False,
        force_quick_gelu=False,
        force_custom_text=False,
        force_patch_dropout=None
    )


try:
    state_dict = torch.load(model_path, map_location='cpu')['state_dict']
    state_dict = {k.replace('module.', ""): v for k, v in state_dict.items()}
    state_dict = {k.replace('backbone.', ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    print("Loaded model from", model_path)
except:
    print("Failed to load model from", model_path)

tokenizer=get_tokenizer(model_cfg)
class EnzymeDataset(Dataset):
    """
    Dataset class for enzyme sequence classification.
    """
    def __init__(self, file_path):
        self.json_data = json.load(open(file_path, 'r'))
        self.sequences = [data['sequence'] for data in self.json_data.values()]
        self.labels = [data['ec'] for data in self.json_data.values()]

    def __len__(self):
        return len(self.sequences)

    def _get_label(self, specific_name):
        labels = [1 if name in specific_name else 0 for name in self.ec_lst]
        return labels

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        labels = self.labels[idx]
        if len(labels) > 1:
            label_convert = ";".join(labels)
        else:
            label_convert = labels[0]
        # print(label_convert)
        return sequence, label_convert

def collate_fn(batch):
    """Define teh collate function for dataloader"""
    protein_code, ec_number = zip(*batch)
    # print(ec_number)
    protein = [torch.from_numpy(one_hot(pts)).float() for pts in protein_code]
    protein_ids = torch.stack(protein)
    ec_total = list()
    for ec_lst in ec_number:
        ec_total.append(tokenizer(ec_lst)[0])
    ec_total_tensor = torch.stack(ec_total)
    output_dict = dict(ec_number=ec_total_tensor, protein_ids=protein_ids)
    return output_dict


train_dataset = EnzymeDataset("/scratch0/zx22/zijie/biotranslator/Demo/data/train_cut.json")
validation_dataset = EnzymeDataset("/scratch0/zx22/zijie/biotranslator/Demo/data/new_cut.json")
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
model = model.to('cpu')
cliploss = ClipLoss()

# for inputs in train_dataloader:
#     ec_number = inputs['ec_number'].to(device)
#     protein_ids = inputs['protein_ids'].to(device)
#     print("ec_number",ec_number.shape)
#     print("protein_ids",protein_ids.shape)
#     break
    
    # prot_feat, text_feat, logit_scale = model(protein_ids, ec_number)
    # total_loss = cliploss(prot_feat, text_feat, logit_scale)
    # print(total_loss)
    
    # num_protein = prot_feat.shape[0]
    # labels = torch.arange(num_protein, device=prot_feat.device)
    
    # logits = logit_scale * prot_feat @ text_feat.T
    # predictions = torch.argmax(logits, dim=1)
    # correct_predictions = torch.sum(predictions == labels)
    # accuracy = correct_predictions.item() / len(labels)
    # print(accuracy)

    
#     break
import json
from tqdm import tqdm
import csv
import pandas as pd


def maximum_separation(dist_lst, first_grad, use_max_grad):
    opt = 0 if first_grad else -1
    gamma = np.append(dist_lst[1:], np.repeat(dist_lst[-1], 10))
    sep_lst = np.abs(dist_lst - np.mean(gamma))
    sep_grad = np.abs(sep_lst[:-1]-sep_lst[1:])
    if use_max_grad:
        # max separation index determined by largest grad
        max_sep_i = np.argmax(sep_grad)
    else:
        # max separation index determined by first or the last grad
        large_grads = np.where(sep_grad > np.mean(sep_grad))
        max_sep_i = large_grads[-1][opt]
    # if no large grad is found, just call first EC
    if max_sep_i >= 5:
        max_sep_i = 0
    return max_sep_i


def write_max_sep_choices(df, csv_name, first_grad=True, use_max_grad=False):
    out_file = open(csv_name + '_maxsep.csv', 'w', newline='')
    csvwriter = csv.writer(out_file, delimiter=',')
    all_test_EC = set()
    for col in df.columns:
        ec = []
        smallest_10_dist_df = df[col].nsmallest(10)
        dist_lst = list(smallest_10_dist_df)
        max_sep_i = maximum_separation(dist_lst, first_grad, use_max_grad)
        for i in range(max_sep_i+1):
            EC_i = smallest_10_dist_df.index[i]
            dist_i = smallest_10_dist_df[i]
            dist_str = "{:.4f}".format(dist_i)
            all_test_EC.add(EC_i)
            ec.append('EC:' + str(EC_i) + '/' + dist_str)
        ec.insert(0, col)
        csvwriter.writerow(ec)
    return


def get_ec_id_dict(filename):
    seq2ec_dict = {}
    ec2seq_dict = {}
    train_data = json.load(open(filename, "r"))
    for ids in list(train_data.keys())[0:32]:
        ec_pair = train_data[ids]
        ec_number = ec_pair['ec']
        protein_sequence = ec_pair['sequence']
        seq2ec_dict[protein_sequence] = ec_number
        for ec in ec_number:
            if ec not in ec2seq_dict.keys():
                ec2seq_dict[ec] = set()
                ec2seq_dict[ec].add(protein_sequence)
            else:
                ec2seq_dict[ec].add(protein_sequence)
    return seq2ec_dict, ec2seq_dict

def get_embedding(seq, seq2ec_dict):
    ec_num = ";".join(seq2ec_dict[seq])
    ec_token = tokenizer(ec_num)[0].unsqueeze(0)
    seq_token = torch.from_numpy(one_hot(seq)).unsqueeze(0).float()
    protein_feat, ec_feat, logit_scale = model(seq_token, ec_token)
    return protein_feat, ec_feat
    
    
def get_train_embedding(seq2ec_dict, ec2seq_dict, device, dtype):
    '''
    Loading esm embedding in the sequence of EC numbers
    prepare for calculating cluster center by EC
    '''
    train_embedding = []
    for ec in list(ec2seq_dict.keys()):
        protein_seq_lst = list(ec2seq_dict[ec])
        protein_embedding_lst = []
        for seq in protein_seq_lst:
            protein_embedding, ec_embedding = get_embedding(seq, seq2ec_dict)
            protein_embedding_lst.append(protein_embedding)
        train_embedding = train_embedding + protein_embedding_lst
    return torch.cat(train_embedding).to(device=device, dtype=dtype)

def get_test_embedding(seq2ec_dict, device, dtype):
    '''
    Instead of loading esm embedding in the sequence of EC numbers
    the test embedding is loaded in the sequence of queries
    then inferenced with model to get model embedding
    '''
    protein_seq_lst = list(seq2ec_dict.keys())
    protein_embedding_lst = []
    for seq in protein_seq_lst:
        protein_embedding, ec_embedding = get_embedding(seq, seq2ec_dict)
        protein_embedding_lst.append(protein_embedding)
        
    test_embedding = torch.cat(protein_embedding_lst).to(device=device, dtype=dtype)
    return test_embedding

def get_cluster_center(train_embedding, ec2seq_dict_train):
    cluster_center_model = {}
    id_counter = 0
    with torch.no_grad():
        for ec in tqdm(list(ec2seq_dict_train.keys())):
            protein_seq_lst = list(ec2seq_dict_train[ec])
            id_counter_prime = id_counter + len(protein_seq_lst)
            embeding_cluster = train_embedding[id_counter: id_counter_prime]
            cluster_center = embeding_cluster.mean(dim=0)
            cluster_center_model[ec] = cluster_center.detach().cpu()
            id_counter = id_counter_prime
    return cluster_center_model

def dist_map_helper(keys1, lookup1, keys2, lookup2):
    dist = {}
    for i, key1 in tqdm(enumerate(keys1)):
        current = lookup1[i].unsqueeze(0)
        dist_norm = (current - lookup2).norm(dim=1, p=2)
        dist_norm = dist_norm.detach().cpu().numpy()
        dist[key1] = {}
        for j, key2 in enumerate(keys2):
            dist[key1][key2] = dist_norm[j]
    return dist

def get_dist_map_test(train_embedding, test_embedding, ec2seq_dict_train, seq2ec_dict_test, device, dtype):
    '''
    Get the pair-wise distance map for test queries and train EC cluster centers
    map is of size of (N_test_ids, N_EC_train)
    '''
 
    cluster_center_model = get_cluster_center(
        train_embedding, ec2seq_dict_train)
    total_ec_n, out_dim = len(ec2seq_dict_train.keys()), train_embedding.size(1)
    model_lookup = torch.zeros(total_ec_n, out_dim, device=device, dtype=dtype)
    ec_lst = list(cluster_center_model.keys())
    for i, ec in enumerate(ec_lst):
        model_lookup[i] = cluster_center_model[ec]
    model_lookup = model_lookup.to(device=device, dtype=dtype)

    seq_test_lat = list(seq2ec_dict_test.keys())
    eval_dist = dist_map_helper(seq_test_lat, test_embedding, ec_lst, model_lookup)
    return eval_dist
    
def max_separation():
    use_cuda = torch.cuda.is_available()
    # device = torch.device("cuda:0" if use_cuda else "cpu")
    device = 'cpu'
    dtype = torch.float32
    out_filename = "/scratch0/zx22/zijie/biotranslator/Demo/data/max_sep"

    seq2ec_dict_train, ec2seq_dict_train = get_ec_id_dict("/scratch0/zx22/zijie/biotranslator/Demo/data/train_cut.json")
    seq2ec_dict_test, ec2seq_dict_test = get_ec_id_dict("/scratch0/zx22/zijie/biotranslator/Demo/data/new_cut.json")

    train_embedding = get_train_embedding(seq2ec_dict_train, ec2seq_dict_train, device, dtype)
    test_embedding = get_test_embedding(seq2ec_dict_test, device, dtype)
    
    eval_dist = get_dist_map_test(train_embedding, test_embedding, ec2seq_dict_train, seq2ec_dict_test, device, dtype)
    eval_df = pd.DataFrame.from_dict(eval_dist)
    
    write_max_sep_choices(eval_df, out_filename)

max_separation()
