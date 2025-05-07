"""
hugging Face上面的Tokenizer
"""

from transformers import AutoTokenizer
from transformers.models.bert import BertTokenizer
from transformers.models.llama import LlamaTokenizer
from pprint import pprint

if __name__ == '__main__':
    # 加载预训练 Tokenizer
    # tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

    # 分词流程
    text = "# 加载预训练 Tokenizer, 非pythorch官网，也不是python"
    text = "深度学习Deep Learning很棒！"
    tokens = tokenizer.tokenize(text)  # 输出：['i', 'love', 'nl', '##p', '!']
    ids = tokenizer.encode(text)  # 输出：[101, 1045, 2293, 17953, 24471, 999, 102]
    decoded = tokenizer.decode(ids)  # 输出："i love nlp!"
    print(tokens)
    print(ids)
    print(decoded)
