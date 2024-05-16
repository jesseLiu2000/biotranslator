import torch
import os
import json
import wandb
import pickle
import numpy as np
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
from sklearn.metrics import (
    accuracy_score,
    hamming_loss,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.utils.class_weight import compute_class_weight


wandb.login(key='471850dc0af0748ea73eb1fbf278a9075c79f11d')
wandb.init()

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


accelerator = Accelerator()
train_dataset = EnzymeDataset("/scratch0/zx22/zijie/biotranslator/Demo/data/train_cut.json")
validation_dataset = EnzymeDataset("/scratch0/zx22/zijie/biotranslator/Demo/data/new_cut.json")


model = accelerator.prepare(model)
train_dataset = accelerator.prepare(train_dataset)
validation_dataset = accelerator.prepare(validation_dataset)

timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

training_args = TrainingArguments(
    output_dir=f"/scratch0/zx22/zijie/biotranslator/Demo/results/{timestamp}",
    num_train_epochs=1,
    learning_rate=1e-03,
    do_eval=False,
    lr_scheduler_type="cosine",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    remove_unused_columns=False,
    warmup_steps=500,
    weight_decay=0.01,
    # evaluation_strategy="epoch",
    save_strategy="epoch",
    # eval_steps=1,
    # load_best_model_at_end=True,
    # metric_for_best_model="f1",
    greater_is_better=True,
    push_to_hub=False,
    logging_dir=None,
    logging_first_step=False,
    logging_steps=200,
    save_total_limit=7,
    no_cuda=False,
    report_to='wandb'
)

# def compute_metrics(p):
#     """Compute metrics for evaluation."""
#     total_feat, logit_scale = p
#     prot_feat, text_feat = total_feat[0], total_feat[1]
#     clip_metric = ClipMetric()
#     acc_protein, acc_ec = clip_metric(prot_feat, text_feat, logit_scale)
#     prob = torch.sigmoid(torch.matmul(prot_feat, text_feat.T))
#     return {"acc_protein": acc_protein, "acc_ec": acc_ec, "prob": prob}

class WeightedTrainer(Trainer):
    def __int__(self, *args, **kwargs):
        super().__int__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        protein, text_ids = inputs['protein_ids'], inputs['ec_number']
        prot_feat, text_feat, logit_scale = model(protein, text_ids)
        logits_per_image = logit_scale * prot_feat @ text_feat.T
        logits_per_text = logit_scale * text_feat @ prot_feat.T
        num_logits = logits_per_image.shape[0]
        
        labels = torch.arange(num_logits, device=device, dtype=torch.long)

        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
            ) / 2
        return total_loss

    # def prediction_step(self, model, inputs, prediction_loss_only=False, ignore_keys=None):
    #     cliploss = ClipLoss()
    #     protein, text_ids = inputs['protein_ids'], inputs['ec_number']
    #     prot_feat, text_feat, logit_scale = model(protein, text_ids)
    #     total_loss = cliploss(prot_feat, text_feat, logit_scale[0])
    #     total_feat = torch.stack((prot_feat, text_feat), dim=0)
    #     return (total_loss.detach(), total_feat.detach(), logit_scale[0].detach())


# Initialize Trainer
trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    #eval_dataset=validation_dataset,
    data_collator=collate_fn,
    # compute_metrics=compute_metrics,
)

# Train and Evaluate the model
trainer.train()
save_path = os.path.join("/scratch0/zx22/zijie/biotranslator/Demo/results/", f"best_model_biotranslator/{timestamp}")
trainer.save_model(save_path)
# tokenizer.save_pretrained(save_path)
model = torch.load(f'/scratch0/zx22/zijie/biotranslator/Demo/results/{timestamp}/model.pt')




