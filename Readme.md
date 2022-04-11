# Huggingface transformer
强烈建议看huggingface-transformer的[Tutorial](https://huggingface.co/course/chapter0/1?fw=pt)，有问题一般可以现在tutorial里面找<br>
[Model Hub](https://huggingface.co/models?sort=downloads) 查询model的名字（model可以直接用在pipeline里）<br>
重点讲tokenizer，bertmodel的：https://zhuanlan.zhihu.com/p/120315111

## Data Processing
dataset hub: https://huggingface.co/datasets<br>
dataset文档：https://huggingface.co/docs/datasets/load_hub<br>
更详细的：包括load local/custom dataset，filter，slice，map，split，见：
https://huggingface.co/course/chapter5/3?fw=pt<br>
By default, loading local files creates a **DatasetDict** object with a train split

```
from dataset import load_dataset
custom_data = load_dataset("csv", data_files="my_file.csv")
```
create a random sample by chaining the Dataset.shuffle() and Dataset.select() 
```
drug_sample = drug_dataset["train"].shuffle(seed=42).select(range(1000))
# Peek at the first few examples
```

## Tokenizer
1. 要知道哪个model用了哪个tokenizer，需要去model_hub里找model对应的tokenizer。<br>
用AutoTokenizer.from_pretrained(model_name)可以自动载入对应的tokenizer。<br>
2. 另一种方法是去Transformer包的model中查找相应的模型和对应的tokenizer，如BertTokenizer,RobertaTokenizer等，可以直接使用。<br>
Tokenizer对文本进行分词并转化为对应的input_id，这里的id是与bert中embedding矩阵的索引号.<br>
**BertTokenizer**只能加载bert的tokenizer，**AutoTokenizer**可以根据名字加载不同的tokenizer<br>
3. Tokenizer中可以指定padding, truncate, 返回类型（tensor，但tensor一定要每个句子长度相等，即padding）<br>
4. Tokenizer的输入应该是str或List of str, 输出是Dictionary,包括 'input_ids': tensor，'token_type_ids': tensor，'attention_mask': tensor<br>
```
# from_pretrained方法可以载入tokenizer或预训练的模型 
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
encoded_input = tokenizer("我是一句话")

model_inputs = tokenizer(sequences, padding="max_length"/True, truncation = True, return_tensors="pt")


```
tokenizer输出的是python Dictionary
 ```
 {'input_ids': [101, 2769, 3221, 671, 1368, 6413, 102], 
'token_type_ids': [0, 0, 0, 0, 0, 0, 0], 
'attention_mask': [1, 1, 1, 1, 1, 1, 1]}
 ```
  **Transformer models only accept tensors as input.** 
 ```
  raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
```
 其中值得一提的是，BERT的vocab预留了不少unused token，如果我们会在文本中使用特殊字符，在vocab中没有，这时候就可以通过替换vacab中的unused token，
 实现对新的token的embedding进行训练。<br>
在transformer库中：models.bert.tokenization_bert 
 ```
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "bert-base-uncased": "https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt",
        "bert-large-uncased": "https://huggingface.co/bert-large-uncased/resolve/main/vocab.txt", }
```
打开第一个地址就能得到bert-base-uncased的vocab信息

 ## Model &Config
1. Model的input一定要是**tensor**, 用torch.tensor(encoded_sequence)转换成tensor。<br>
2. 在[Model hub](https://huggingface.co/models?sort=downloads)查到想要用的model的名字后，通过AutoModel.from_pretrained(model_name)就能载入对应model。<br>
也可以自己去Transformer库中的models找。例如bert.modeling_bert.py 中提供了**BertModel**等不同的预训练模型以供下载。并包含了BertEmbeddings，BertEncoder，BertPooler等的实现，可以按需修改模型结构。<br>
**BertEmbeddings**这个类中可以清楚的看到，embedding由三种embedding相加得到，经过layernorm 和 dropout后输出。<br>
**BertEncoder**主要将embedding的输出，逐个经过每一层Bertlayer的处理，得到各层hidden_state，再根据**config**的参数，来决定最后是否所有的hidden_state都要输出。<br>
**Bertpooler** 其实就是将BERT的[CLS]的hidden_state 取出，经过一层DNN和Tanh计算后输出。<br>
在这个文件中还有上述基础的**BertModel**的进一步的变化，比如**BertForMaskedLM，BertForNextSentencePrediction**这些是Bert加了预训练头的模型，还有**BertForSequenceClassification， BertForQuestionAnswering** 这些加上了特定任务头的模型。<br>

重点看下BertModel：
```
from transformers import BertConfig, BertModel
# Building the config
config = BertConfig()

# Building the model from the config
# ! In this case model weights are randomly initialized!
model = BertModel(config)

# model weights are loaded from pretrained model. recommend using AutoModel here for checkpoint-agnostic code
model = BertModel.from_pretrained()
output = model(**model_inputs)
```

```
class BertModel(BertPreTrainedModel):
class BertPreTrainedModel(PreTrainedModel):
class PreTrainedModel(nn.Module, ModuleUtilsMixin, GenerationMixin, PushToHubMixin):
```
因此BertModel是PyTorch的nn.Module和BertPreTrainedModel的subclass。<br>
BertModel的类定义中，由embedding，encoder，pooler组成，forward时顺序经过三个模块，输出output。
```python
class BertModel(BertPreTrainedModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None

        self.init_weights()
        
     def forward(
        self, input_ids=None, attention_mask=None, token_type_ids=None,
        position_ids=None, head_mask=None, inputs_embeds=None,
        encoder_hidden_states=None, encoder_attention_mask=None,
    ):
    """ 省略部分代码 """
    
        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        outputs = (sequence_output, pooled_output,) + encoder_outputs[
            1:
        ]  # add hidden_states and attentions if they are here
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)

```

BertModel里有forward()方法，forward()方法中实现了将Token转化为词向量，再将词向量进行多层的Transformer Encoder的复杂变换。<br>
forward()方法的入参有input_ids、attention_mask、token_type_ids等等，实际上就是刚才Tokenizer部分的输出。<br>
 ```
 model = BertModel.from_pretrained("bert-base-chinese")
 bert_output = model(input_ids=batch['input_ids'])
 # len(bert_output) = 2
 ```
通过model(x)自动调用forward方法，返回模型预测的结果，返回结果是一个tuple(torch.FloatTensor)，即多个Tensor组成的tuple。tuple默认返回**两个**重要的Tensor：<br>
**last_hidden_state**：输出序列每个位置的语义向量，形状为：(batch_size, sequence_length, hidden_size)。<br>
**pooler_output**：[CLS]符号对应的语义向量，经过了全连接层和tanh激活；该向量可用于下游分类任务。<br>
用outputs["last_hidden_state"])或outputs[0]来访问。<br>

上述BertModel的output是一个hidden-layer dense-vector，如果要用在textclassification任务中，要么自己在BertModel上再加linear layer做fine-tune；要么可以直接用训练过的BertForSequenceClassification，返回的是logits（注意任何模型返回的要么是dense vector要么是logits），还要自己再加上torch.nn.functional.softmax(outputs.logits, dim=-1)才能变成prediction。对于输出的label可以去model.config.id2label看。

## 修改config信息
在transformer库中： models.bert.configuration_bert<br>
```
BERT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "bert-base-uncased": "https://huggingface.co/bert-base-uncased/resolve/main/config.json",
    "bert-large-uncased": "https://huggingface.co/bert-large-uncased/resolve/main/config.json",}
```
打开第一个地址就能得到bert-base-uncased的模型的配置信息
```
{
  "architectures": [
    "BertForMaskedLM"
  ],
  "attention_probs_dropout_prob": 0.1,
  "gradient_checkpointing": false,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "model_type": "bert",
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "pad_token_id": 0,
  "position_embedding_type": "absolute",
  "transformers_version": "4.6.0.dev0",
  "type_vocab_size": 2,
  "use_cache": true,
  "vocab_size": 30522
}
```
## Fine-tune
1. BERT可以进行很多下游任务，transformers库中实现了一些下游任务。比如单文本分类，transformers库提供了BertForSequenceClassification类。<br>
#AutoModelForSequenceClassification可以根据模型名称用bert以外的模型。<br>
想知道transformer提供了哪些model，可以去transformer库的init文件里看，比如from model.bert import BertForSequenceClassification,<br>
然后再看模型继承的父类，就能看懂和修改模型结构。我们也可以参考transformers中的实现，来做自己想做的任务。<br>
2. 官方fine-tune的example： https://huggingface.co/docs/transformers/custom_datasets <br>
3. 模型的训练有两种方法，一种是用huggingface自带的Trainer方法，一种是Pytorch。

### 方法一： Trainer API<br>
**1. TrainingArguments class** <br>
https://huggingface.co/course/chapter3/3?fw=pt

### 方法二： Pytorch<br>
1. 用Dataloader产生batch数据（Trainer API不用这样做）
```
from torch.utils.data import DataLoader

train_dataloader = DataLoader(
    tokenized_datasets["train"], shuffle=True, batch_size=8, collate_fn=data_collator
)
```
2. 定义模型
```
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
outputs = model(**batch)
print(outputs.loss, outputs.logits.shape)
```
3. 定义optimizer，learning rate，epoch
```
from transformers import AdamW

optimizer = AdamW(model.parameters(), lr=5e-5)
```
4. 训练模型
```
from tqdm.auto import tqdm

progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step() # 更新learning rate的
        optimizer.zero_grad()
        progress_bar.update(1)
```
5. 把模型在eval_dataloader上评估：
```
from datasets import load_metric

metric = load_metric("glue", "mrpc")
model.eval()
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

metric.compute()
```

## multiple GPU acceleration
https://huggingface.co/course/chapter3/4?fw=pt

## Saving model
```
model.save_pretrained("directory_on_my_computer")
```
config.json & pytorch_model.bin

## pipeline
currently available [pipelines](https://huggingface.co/docs/transformers/main_classes/pipelines)<br>
pipeline()相当于封装了tokenizer和model，只要输入text，就能得到特定任务的答案: pipeline() accepts any model from the Model Hub.  <br>

```
from transformers import pipeline
classifier = pipeline(task = "sentiment-analysis", model = xxx)
# defaulted to distilbert-base-uncased-finetuned-sst-2-english
classifier("I've been waiting for a HuggingFace course my whole life.")
```
也可以指定pipeline中的tokenizer和model
```
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2")

# generator是一个class：transformers.pipelines.text_generation.TextGenerationPipeline
generator = pipeline(task="text-generation", model=model, tokenizer=tokenizer)
```

