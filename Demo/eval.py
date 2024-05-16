import torch
import numpy as np
from open_clip.utils import one_hot
from open_clip import create_model_and_transforms, get_tokenizer


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

protein = 'MAFSAEDVLKEYDRRRRMEALLLSLYYPNDRKLLDYKEWSPPRVQVECPKAPVEWNNPPSEKGLIVGHFSGIKYKGEKAQASEVDVNKMCCWVSKFKDAMRRYQGIQTCKIPGKVLSDLDAKIKAYNLTVEGVEGFVRYSRVTKQHVAAFLKELRHSKQYENVNLIHYILTDKRVDIQHLEKDLVKDFKALVESAHRMRQGHMINVKYILYQLLKKHGHGPDGPDILTVKTGSKGVLYDDSFRKIYTDLGWKFTPL'
# encode protein using one-hot encoding
protein = torch.from_numpy(one_hot(protein)).cuda().unsqueeze(0).float()
# tokenize text
text = 'Putative transcription factor', 'Putative transcription factor'
text_ids = tokenizer(text)[0].unsqueeze(0).cuda()
with torch.no_grad():
    prot_feat, text_feat, _ = model(protein, text_ids)
    prob = torch.sigmoid(torch.matmul(prot_feat, text_feat.T))
print("The calculated probability is", prob.item())
