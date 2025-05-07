import peft
import torch
import torch.nn as nn
from my_bert_model import BertForSeqCls


class Bert_LoRA:
    pass


if __name__ == '__main__':
    lora_config = peft.LoraConfig(
        r=16,
        target_modules=['my_cls'],
        lora_alpha=32
    )

    in_x = torch.randint(low=0, high=1024, size=(6, 32))  # [batch_size, seq_length]
    base_model_name = 'bert-base-chinese'
    my_model = BertForSeqCls(num_labels=2, base_model_name_or_path=base_model_name)

    peft_model = peft.get_peft_model(my_model, lora_config)
    # print(peft_model)
    print(peft_model.print_trainable_parameters())
