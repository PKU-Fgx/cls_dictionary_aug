from configuration import Config
from dataprocesser import myDataset, myFn
from torch.utils.data import DataLoader
from model import myModel
from transformers import AutoConfig, AutoTokenizer
from utils import Trainer

# ===================
#    1. 配置文件
# ===================
config = Config()

# ===================
#    2. 数据处理
# ===================
tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model_path, use_fast=True)

train_dataset = myDataset(
    config,
    tokenizer = tokenizer
)
train_dataloader = DataLoader(
    train_dataset,
    batch_size = config.bs,
    shuffle = False,
    collate_fn = lambda batch: myFn(batch, tokenizer, config)
)

# ===================
#    3. 模型
# ===================
model_config = AutoConfig.from_pretrained(config.pretrained_model_path)
model = myModel(model_config, config).to(config.device)

# ===================
#    4. 开训
# ===================
trainer = Trainer(model, train_dataloader, config)
trainer.train()