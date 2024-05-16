import torch
import os
import json
import wandb
import pickle
import numpy as np
import torchmetrics
import torch.nn as nn
import torch.nn.functional as F
from open_clip.utils import one_hot
from datetime import datetime
from accelerate import Accelerator
from open_clip import create_model_and_transforms, get_tokenizer, ClipLoss, ClipMetric
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import (
    TrainingArguments,
    Trainer
)
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import precision_score, recall_score, roc_auc_score, accuracy_score, f1_score, average_precision_score

from sklearn.utils.class_weight import compute_class_weight

from open_clip.utils import one_hot
from open_clip import create_model_and_transforms, get_tokenizer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

tokenizer=get_tokenizer(model_cfg)
model_path = "/scratch0/zx22/zijie/biotranslator/Demo/results/model_ckpt_70/4_model.pt"
    
model = torch.load(model_path)
model.eval()

# model_path = "epoch_64.pt"

# model_cfg = 'PubMedBERT_512-CNN'
# ### create and load model ###
# model, _, _ = create_model_and_transforms(
#         model_cfg,
#         '',
#         precision='amp',
#         device=torch.device('cuda'),
#         jit=False,
#         force_quick_gelu=False,
#         force_custom_text=False,
#         force_patch_dropout=None
#     )

# try:
#     state_dict = torch.load(model_path, map_location='cpu')['state_dict']
#     state_dict = {k.replace('module.', ""): v for k, v in state_dict.items()}
#     state_dict = {k.replace('backbone.', ""): v for k, v in state_dict.items()}
#     model.load_state_dict(state_dict, strict=False)
#     print("Loaded model from", model_path)
# except:
#     print("Failed to load model from", model_path)
# print(model)

def maximum_seperation(dist_lst, first_grad, use_max_grad, dist_idxs, n_text):
    opt = 0 if first_grad else -1
    multi_preds = []
    for i in range(len(dist_lst)):
        dist = dist_lst[i].cpu().numpy()
        gamma = np.append(dist[1:], np.repeat(dist[-1], 10))
        sep_lst = np.abs(dist - np.mean(gamma))
        sep_grad = np.abs(sep_lst[:-1]-sep_lst[1:])
        if use_max_grad:
            max_sep_i = np.argmax(sep_grad)
        else:
            large_grads = np.where(sep_grad > np.mean(sep_grad))
            if len(large_grads[-1]) == 0:
                max_sep_i = 0
            else:
                max_sep_i = large_grads[-1][opt]
        if max_sep_i >= 3:
            max_sep_i = 0
        multi_pred = None
        for j in range(max_sep_i + 1):
            EC_i = dist_idxs[i][j]
            EC_tensor = EC_i
            if multi_pred is None:
                multi_pred = F.one_hot(EC_tensor, num_classes = n_text)
            else:
                multi_pred = multi_pred + F.one_hot(EC_tensor, num_classes = n_text)
        multi_preds.append(multi_pred)
    return torch.stack(multi_preds)

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
            label_convert = labels
        return sequence, label_convert

def collate_fn(batch):
    """Define teh collate function for dataloader"""
    protein_code, ec_number = zip(*batch)
    protein = [torch.from_numpy(one_hot(pts)).float() for pts in protein_code]
    protein_ids = torch.stack(protein)
    ec_total = list()
    for ec_lst in ec_number:
        ec_total.append(tokenizer(ec_lst).squeeze())
    ec_total_tensor = torch.stack(ec_total)
    output_dict = dict(ec_number=ec_total_tensor, protein_ids=protein_ids)
    return output_dict


def cal_acc(labels_np, preds_cpu):
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import label_binarize

    y_true_flat = labels_np.flatten()
    y_scores_flat = preds_cpu.flatten()

    # Convert to NumPy arrays
    y_true_np = y_true_flat.numpy()
    y_scores_np = y_scores_flat.numpy()
    

    # Calculate AUROC
    if np.unique(y_true_np).tolist() == [0, 1]:
        # Calculate AUROC
        auc_score = roc_auc_score(y_true_np, y_scores_np)
        # acc = accuracy_score(y_true_np, y_scores_np)
        print("AUROC Score:", auc_score)
        # print("acc is", acc)
    else:
        print("Error: y_true must contain only binary labels (0 or 1).")
    # rows_equal = (labels_np == preds_cpu).all(dim=1)
    # print(rows_equal)
    # tensor2 = torch.zeros(392, 392)
    # is_zero_tensor = (preds_cpu == 0).all()
    # non_zero_indices = preds_cpu.nonzero()
    # print(is_zero_tensor)
    # print(non_zero_indices)
    # print(labels_np.shape)
    # print(labels_np)
    # print(preds_cpu.shape)
    # print(preds_cpu)

accelerator = Accelerator()
train_dataset = EnzymeDataset("/scratch0/zx22/zijie/biotranslator/Demo/data/train_cut.json")
validation_dataset = EnzymeDataset("/scratch0/zx22/zijie/esm_ft/data/new_cut.json")
model = accelerator.prepare(model)
train_dataset = accelerator.prepare(train_dataset)
validation_dataset = accelerator.prepare(validation_dataset)
validation_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
clip_loss = ClipLoss()

labels = []
preds = []
feature_lst = []
n_text = len(validation_dataloader)
max_sep_topk = int(n_text / 2)
with torch.no_grad():
    for idx, inputs in enumerate(validation_dataloader):
        ec_number = inputs['ec_number'].to(device)
        protein_ids = inputs['protein_ids'].to(device)

        prot_feat, text_feat, logit_scale = model(protein_ids, ec_number)
        prob = torch.sigmoid(torch.matmul(prot_feat, text_feat.T))
        feature_lst.append(text_feat) 
        
    ec_tensor = torch.concat(feature_lst)
        
with torch.no_grad():
    for idx, inputs in enumerate(validation_dataloader):
        ec_number = inputs['ec_number'].to(device)
        protein_ids = inputs['protein_ids'].to(device)
        
        prot_feat, _, _ = model(protein_ids, ec_number)
        label = F.one_hot(torch.tensor(idx), num_classes=n_text)
        
        dot_similarity_text = prot_feat @ ec_tensor.T
        dot_similarity_text = F.softmax(dot_similarity_text, dim=1)
        # dot_similarity_text = torch.sigmoid(dot_similarity_text)
        pred = dot_similarity_text
        # print(pred)
        # smallest_k_dist, smallest_k_idxs = dot_similarity_text.topk(max_sep_topk, dim=-1)
        # print("smallest_k_dist, smallest_k_idxs", smallest_k_dist, smallest_k_idxs)
        # pred = maximum_seperation(smallest_k_dist, True, False, smallest_k_idxs, n_text)
        # print("label", label)
        # print("pred", pred)
        labels.append(label)
        preds.append(pred)
        # print("preds", preds)
        
        
    
    preds_np = torch.concat(preds)
    labels_np = torch.stack(labels)
    # print(preds_np)
    # print(labels_np)
    
    preds_cpu = preds_np.cpu()
    labels_np = labels_np.cpu()
    # print("labels_np", labels_np)
    # print("preds_cpu", preds_cpu)

    # pp_f1 = f1_score(labels_np, preds_cpu, average='weighted', zero_division=0)
    # pp_recall = recall_score(labels_np, preds_cpu, average='weighted', zero_division=0)
    # pp_precision = precision_score(labels_np, preds_cpu, average='weighted', zero_division=0)
    acc = cal_acc(labels_np, preds_cpu)
    
    # metric_dict = {"F1": pp_f1, "Recall": pp_recall, "Precision": pp_precision}
    # print(metric_dict) 

"""
{'F1': 0.018648023119022045, 'Recall': 0.04336734693877551, 'Precision': 0.013683390022675736}
"""