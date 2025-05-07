import torch.nn
from torch import nn
from transformers import BertModel


__all__ = [
    'BertForSeqCls'
]

class BertForSeqCls(nn.Module):
    def __init__(self, num_labels, base_model_name_or_path='bert-base-chinese', dropout_rate=0.05):
        super(BertForSeqCls, self).__init__()
        self.bert = BertModel.from_pretrained(base_model_name_or_path)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.my_cls = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, ipt_idx, attention_mask=None, token_type_ids=None):
        out = self.bert(ipt_idx)
        """
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None
        """

        # 取[CLS]位置的输出向量
        cls_output = out.last_hidden_state[:, 0, :]  # shape: (batch_size, hidden_size)
        # Dropout + 分类层
        cls_output = self.dropout(cls_output)
        logits = self.my_cls(cls_output)
        return logits


if __name__ == '__main__':
    in_x = torch.randint(low=0, high=1024, size=(6, 32))  # [batch_size, seq_length]

    base_model_name = 'bert-base-chinese'
    my_model = BertForSeqCls(num_labels=2, base_model_name_or_path=base_model_name)
    res = my_model(in_x)
    print(res.shape)  # p[batch_size, cls_num]
    print(res)

