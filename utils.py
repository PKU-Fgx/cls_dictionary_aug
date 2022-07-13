import logging

from tqdm import tqdm
from transformers import AdamW, get_cosine_schedule_with_warmup

class Trainer:
    
    def __init__(self, model, train_dataloader, config):
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        
        self.logger = self.set_logger()
        
        self.optimizer = AdamW(self.model.parameters(), lr=self.config.lr)
        self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, 0, len(self.train_dataloader) * self.config.epoches)
        
        self.training_info = "Training Epoch[ {} | {} ] AvgLoss = {:.8f}"
    
    def train(self):
        step = 0
        for epoch in range(self.config.epoches):
            self.model.train()
            loss_list = list()
            train_bar = tqdm(self.train_dataloader)
            for sentence_encoded, augged_info_idx, augged_info_input_ids, mlm_input_encoded, mlm_labels in train_bar:
                self.optimizer.zero_grad()
                loss = self.model(sentence_encoded, augged_info_idx, augged_info_input_ids, mlm_input_encoded, mlm_labels)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                
                loss_list.append(loss.item())
                avg_loss = sum(loss_list) / len(loss_list)
                train_bar.set_description(self.training_info.format(epoch, self.config.epoches, avg_loss))
                
                if step % (int(len(self.train_dataloader) * self.config.epoches / 500)) == 0:
                    self.logger.info(self.training_info.format(epoch, self.config.epoches, avg_loss))
                    self.model.save_pretrained(self.config.save_path)
                step += 1

    def set_logger(self):
        logger = logging.getLogger(__name__)
        logger.setLevel(level = logging.INFO)
        
        handler = logging.FileHandler(self.config.logger_path, mode="w")
        handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        
        logger.addHandler(handler)
        
        return logger