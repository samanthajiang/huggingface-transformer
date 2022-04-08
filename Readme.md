# Huggingface transformer
强烈建议看huggingface-transformer的[Tutorial](https://huggingface.co/course/chapter0/1?fw=pt)，有问题一般可以现在tutorial里面找<br>
[Model Hub](https://huggingface.co/models?sort=downloads) 查询model的名字（model可以直接用在pipeline里）<br>
重点讲tokenizer，bertmodel的：https://zhuanlan.zhihu.com/p/120315111

## Data Processing
dataset文档：https://huggingface.co/docs/datasets/load_hub<br>
更详细的：包括load local/custom dataset，filter，slice，map，split，见：
https://huggingface.co/course/chapter5/3?fw=pt<br>
By default, loading local files creates a **DatasetDict** object with a train split

```
from dataset import load_dataset
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
**BertTokenizer**只能加载bert的tokenizer，**AutoTokenizer**可以根据名字加载不同的tokenizer

```
# from_pretrained方法可以载入tokenizer或预训练的模型 
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

 ## Model

transformer库中：models.bert.modeling_bert.py 提供了不同的预训练模型以供下载。并包含了BertEmbeddings，BertEncoder，BertPooler等的实现，可以按需修改。<br>
**BertEmbeddings**这个类中可以清楚的看到，embedding由三种embedding相加得到，经过layernorm 和 dropout后输出。<br>
**BertEncoder**主要将embedding的输出，逐个经过每一层Bertlayer的处理，得到各层hidden_state，再根据**config**的参数，来决定最后是否所有的hidden_state都要输出。<br>
**Bertpooler** 其实就是将BERT的[CLS]的hidden_state 取出，经过一层DNN和Tanh计算后输出。<br>
在这个文件中还有上述基础的BertModel的进一步的变化，比如**BertForMaskedLM，BertForNextSentencePrediction**这些是Bert加了预训练头的模型，还有**BertForSequenceClassification， BertForQuestionAnswering** 这些加上了特定任务头的模型。<br>


重点看下BertModel：
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
## 下游任务
BERT可以进行很多下游任务，transformers库中实现了一些下游任务。比如单文本分类，transformers库提供了BertForSequenceClassification类。<br>
#AutoModelForSequenceClassification可以根据模型名称用bert以外的模型。<br>
想知道transformer提供了哪些model，可以去transformer库的init文件里看，比如from model.bert import BertForSequenceClassification,<br>
然后再看模型继承的父类，就能看懂和修改模型结构。<br>
我们也可以参考transformers中的实现，来做自己想做的任务。


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
