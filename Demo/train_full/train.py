import torch
import os
import json
import wandb
from tqdm import tqdm
# import clip
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, optim
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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

### load the tokenizer ###
tokenizer=get_tokenizer(model_cfg)
# for pn, p in model.named_parameters():
#     print(pn, p)


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

    # def _get_label(self, specific_name):
    #     labels = [1 if name in specific_name else 0 for name in self.ec_lst]
    #     return labels

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

def convert_models_to_fp32(model): 
    for pn, p in model.named_parameters(): 
        if p.grad is not None:
            p.data = p.data.float() 
            p.grad.data = p.grad.data.float() 

train_dataset = EnzymeDataset("/scratch0/zx22/zijie/biotranslator/Demo/train_full/data/train_70.json")
validation_dataset = EnzymeDataset("/scratch0/zx22/zijie/biotranslator/Demo/data/new_cut.json")
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)
validation_dataloader = DataLoader(validation_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')


model.to(device)
model.train()
cliploss = ClipLoss()
clipmetric = ClipMetric()

#https://github.com/openai/CLIP/issues/57
def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        if p.grad is not None:
            p.data = p.data.float() 
            p.grad.data = p.grad.data.float() 

if device == "cpu":
    model.float()

EPOCH = 10
BATCH_SIZE = 32 # switch 16 -> 32
loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=5e-5,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2)
optimizer = optim.Adam(model.parameters(), lr=1e-5)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_dataloader)*EPOCH)


best_te_loss = 1e5
best_ep = -1
for epoch in range(EPOCH):
    print(f"running epoch {epoch}, best test loss {best_te_loss} after epoch {best_ep}")
    step = 0
    tr_loss = 0
    model.train()
    pbar = tqdm(train_dataloader, leave=False)
    for inputs in pbar:
        step += 1
        optimizer.zero_grad()

        ec_number = inputs['ec_number'].to(device)
        protein_ids = inputs['protein_ids'].to(device)

        image_features, text_features, logit_scale = model(protein_ids, ec_number)
        
        prob = torch.sigmoid(torch.matmul(image_features, text_features.T))
        
        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text = logit_scale * text_features @ image_features.T
        num_logits = logits_per_image.shape[0]
        
        labels = torch.arange(num_logits, device=device, dtype=torch.long)

        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
            ) / 2
        
        total_loss.backward()
        tr_loss += total_loss.item()
        
        optimizer.step()
        scheduler.step()
        # if device == "cpu":
        #     optimizer.step()
        #     scheduler.step()
        # else:
        #     convert_models_to_fp32(model)
        #     optimizer.step()
        #     scheduler.step()
        #     clip.model.convert_weights(model)
        pbar.set_description(f"train batchCE: {total_loss.item()}", refresh=True)
    tr_loss /= step
   
    # torch.save(model.state_dict(), f'/scratch0/zx22/zijie/biotranslator/Demo/results/model_ckpt/{epoch}_model.pth')
    torch.save(model, f'/scratch0/zx22/zijie/biotranslator/Demo/train_full/results/model_ckpt_70/{epoch}_model.pt')
    print("------------------------------------------------------------------")
    print(f"Epoch {epoch}/{EPOCH}, Loss: {total_loss.item():.4f}, Probability: {prob}")
    print("------------------------------------------------------------------")
    

# CUDA_VISIBLE_DEVICES=4 nohup python train.py > log/bio_70.log 2>&1 &
