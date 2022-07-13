from dataclasses import dataclass, field

@dataclass
class Config:
    train_path: str = field(
        default = "/tf/FangGexiang/2.SememeV2/pretrained_data/augged_data_all.json",
        # default = "/tf/FangGexiang/1.SememeV1/myData/augment/augged_data_1.json",
        metadata = { "help": "训练数据位置" }
    )
    pretrained_model_path: str = field(
        default = "/tf/FangGexiang/2.SememeV2/pretrained_model/bert-base-chinese",
        metadata = { "help": "预训练模型位置" }
    )
    bs: int = field(
        default = 8,
        metadata = { "help": "批处理大小" }
    )
    max_length: int = field(
        default = 512,
        metadata = { "help": "每句最长是多少" }
    )
    temp: float = field(
        default = 0.05,
        metadata = { "help": "对比学习的温度系数" }
    )
    device: str = field(
        default = "cuda:2",
        metadata = { "help": "训练位置" }
    )
    lr: float = field(
        default = 1e-5,
        metadata = { "help": "学习率" }
    )
    epoches: int = field(
        default = 2,
        metadata = { "help": "迭代次数" }
    )
    logger_path: str = field(
        default = "./training_log.txt",
        metadata = { "help": "日志打印位置" }
    )
    save_path: str = field(
        default = "../model_saved_v1/",
        metadata = { "help": "模型保存位置" }
    )
    mlm_weight: float = field(
        default = 0.1,
        metadata = { "help": "mlm损失所占比重" }
    )