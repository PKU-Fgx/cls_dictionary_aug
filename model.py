import torch

from torch import nn
from transformers import PreTrainedModel
from transformers.models.bert.modeling_bert import BertOnlyMLMHead
from Bert_Mutil_Replaced import myBertModel

class Similarity(nn.Module):

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp

class myModel(PreTrainedModel):
    
    def __init__(self, config=None, model_args=None):
        super(myModel, self).__init__(config)
        self.config = config
        self.model_args = model_args
        
        self.encoder = myBertModel.from_pretrained(self.model_args.pretrained_model_path)
        self.cls = BertOnlyMLMHead(self.config)
        self.sim = Similarity(self.model_args.temp)

    def forward(self, sentence_encoded, augged_info_idx, augged_info_input_ids, mlm_input_encoded, mlm_labels):
        # ==========================
        # 1. 对比的损失: CL_loss
        # ==========================
        ori_text_pooler_outs = self.encoder(**sentence_encoded).pooler_output  # shape: (batch_size, hidden_size)
        
        if augged_info_idx is not None:
            explain_pooler_outs = self.encoder(**augged_info_input_ids).pooler_output  # shape: (num_sememes, hidden_size)

            sentence_encoded.update({
                "explain_pooler_outs": explain_pooler_outs,
                "batch_index": augged_info_idx
            })
            exp_text_pooler_outs = self.encoder(**sentence_encoded).pooler_output  # shape: (batch_size, hidden_size)
        else:
            exp_text_pooler_outs = self.encoder(**sentence_encoded).pooler_output  # shape: (batch_size, hidden_size)
        
        z1, z2 = ori_text_pooler_outs.unsqueeze(1), exp_text_pooler_outs.unsqueeze(0)
        cos_sim = self.sim(z1, z2)
        
        labels = torch.arange(cos_sim.size(0)).long().to(self.model_args.device)
        loss_fct = nn.CrossEntropyLoss()
        
        CL_loss = loss_fct(cos_sim, labels)
        
        # ==========================
        # 2. MLM的损失: mlm_loss
        # ==========================
        sequence_output = self.encoder(**mlm_input_encoded).last_hidden_state
        prediction_scores = self.cls(sequence_output)

        mlm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), mlm_labels.view(-1))
        
        return CL_loss + self.model_args.mlm_weight * mlm_loss