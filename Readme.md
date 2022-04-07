# huggingface transformer

## tokenizer
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
