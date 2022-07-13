import json
import torch

from tqdm import tqdm
from torch.utils.data import Dataset
from transformers.data.data_collator import DataCollatorForLanguageModeling

class myDataset(Dataset):
    
    def __init__(self, config, **args):
        
        self.config = config
        self.tokenizer = args["tokenizer"]
        
        self.data = list()
        raw_data = json.load(open(self.config.train_path, "r"))
        for data_item in tqdm(raw_data):
            self.data.append([
                data_item["sentence"].split(), data_item["info"]
            ])
        
    def __getitem__(self, idx):
        return self.data[idx]
    
    def __len__(self):
        return len(self.data)

def myFn(batch, tokenizer, config):
    sentence_list, augged_info_list = list(map(list, list(zip(*batch))))
    
    datacollecter = DataCollatorForLanguageModeling(tokenizer)
    
    sentence_encoded = tokenizer(
        sentence_list,
        max_length = config.max_length,
        is_split_into_words = True,
        truncation = True,
        padding = True,
        return_tensors = "pt"
    )
    
    mlm_input_ids, mlm_labels = datacollecter.torch_mask_tokens(sentence_encoded["input_ids"])

    mlm_input_encoded = {
        "input_ids": mlm_input_ids,
        "token_type_ids": sentence_encoded["token_type_ids"],
        "attention_mask": sentence_encoded["attention_mask"]
    }
    
    sentence_encoded = { k: v.to(config.device) for k, v in sentence_encoded.items() }
    mlm_input_encoded = { k: v.to(config.device) for k, v in mlm_input_encoded.items() }
    
    if not any(augged_info_list):
        return [sentence_encoded, None, None, mlm_input_encoded, mlm_labels.to(config.device)]
    
    augged_info_idx, augged_info_input_ids = list(), list()
    
    for i, augged_info in enumerate(augged_info_list):
        for info in augged_info:
            augged_info_idx.append([i, info[0]])
            augged_info_input_ids.append(info[1])

    augged_info_input_ids = tokenizer(
        augged_info_input_ids,
        max_length = config.max_length,
        truncation = True,
        padding = True,
        return_tensors = "pt"
    )
    augged_info_input_ids = { k: v.to(config.device) for k, v in augged_info_input_ids.items() }
    
    return [
        sentence_encoded,
        torch.tensor(augged_info_idx).to(config.device),
        augged_info_input_ids,
        mlm_input_encoded,
        mlm_labels.to(config.device)
    ]