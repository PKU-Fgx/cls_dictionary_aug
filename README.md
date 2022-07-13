## cls_dictionary_aug
### 思路介绍
主要就是将一段文本中的某些在字典中有释义的token用其解释的`cls`(`pooler`)向量去替换其自身的向量然后做对比学习
### 文件介绍
1. `main.py`: 入口函数
2. `configuration.py`: 配置文件
3. `dataprocesser.py`: 处理dataset和collate_fn的
4. `utils.py`: 主要是trainer的定义
5. `model.py`: 模型，基于Bert_Mutil_Replaced去做对比学习，并加上了mlm的损失
6. `Bert_Mutil_Replaced.py`: hugging face的Bert模型，稍微修改了一下self-attention内容
