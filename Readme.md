# Huggingface transformer

## Tokenizer
Transformer中封装了常见的bert模型使用的分词器，如BertTokenizer,RobertaTokenizer等，可以直接使用。<br>
对文本进行分词并转化为对应的input_id，这里的id是与bert中embedding矩阵的索引号.<br>
**BertTokenizer**只能加载bert的tokenizer，**AutoTokenizer**可以根据名字加载不同的tokenizer

```
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
encoded_input = tokenizer("我是一句话")

```
tokenizer输出的是python dictionary
 ```
 {'input_ids': [101, 2769, 3221, 671, 1368, 6413, 102], 
'token_type_ids': [0, 0, 0, 0, 0, 0, 0], 
'attention_mask': [1, 1, 1, 1, 1, 1, 1]}
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

 ## Model
transformer库中：models.bert.modeling_bert.py 提供了不同的预训练模型以供下载。并包含了bert embedding，outputlayer等的实现，可以按需修改。<br>
重点看下BertModel：
```
class BertModel(BertPreTrainedModel):
class BertPreTrainedModel(PreTrainedModel):
class PreTrainedModel(nn.Module, ModuleUtilsMixin, GenerationMixin, PushToHubMixin):
```
因此BertModel是PyTorch的nn.Module和BertPreTrainedModel的subclass。<br>
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

## 下游任务
BERT可以进行很多下游任务，transformers库中实现了一些下游任务。比如单文本分类，transformers库提供了BertForSequenceClassification类。<br>
#AutoModelForSequenceClassification可以根据模型名称用bert以外的模型。<br>
想知道transformer提供了哪些model，可以去transformer库的init文件里看，比如from model.bert import BertForSequenceClassification,<br>
然后再看模型继承的父类，就能看懂和修改模型结构。<br>
我们也可以参考transformers中的实现，来做自己想做的任务。



## 修改模型配置
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

## pipeline
The pipeline() accepts any model from the Model Hub.  <br>
from_pretrained方法可以载入tokenizer或预训练的模型 <br>
tokenizer，model和generator都是class <br>
```
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2")

# generator是一个class：transformers.pipelines.text_generation.TextGenerationPipeline
generator = pipeline(task="text-generation", model=model, tokenizer=tokenizer)
```
## Preproces
### Load a pretrained tokenizer/model<br>
The from_pretrained method lets you quickly load a pretrained model for any architecture.<br>
AutoClasses does this job for you so that you automatically retrieve the relevant model given the name/path to the pretrained weights/config/vocabulary.<br>
Instantiating one of **AutoConfig**, **AutoMode**l, and **AutoTokenizer** will directly create a class of the relevant architecture.For instance: <br>
```
model = AutoModel.from_pretrained("bert-base-cased")
```
will create a model that is an instance of BertModel.<br>
看AutoTokenizer到底支持什么模型(https://huggingface.co/docs/transformers/v4.17.0/en/model_doc/auto#transformers.AutoTokenizer.from_pretrained)
```
from transformers import AutoTokenizer
// Tokenizer的输入应该是str或List of str, 输出是Dictionary,包括 'input_ids': tensor，'token_type_ids': tensor，'attention_mask': tensor
// 另外可以加上 padding=True，truncation=True
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
encoded_input = tokenizer("Do not meddle in the affairs of wizards, for they are subtle and quick to anger.")
tokenizer.decode(encoded_input["input_ids"])
```
AutoModelFor classes let you load a pretrained model for a given task <br>
这样就不用再去修改模型结构而可以直接用了
```
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")
```

## Fine-tune a pretrained model
```
from transformers import AutoTokenizer
from datasets import load_dataset

dataset = load_dataset("yelp_review_full")
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
```
Fine-tune with Trainer<br>
```
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)
```
Then fine-tune your model by calling train():<br>
```
trainer.train()
```
