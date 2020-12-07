import torch

from torch import nn
from transformers import BertModel, BertForSequenceClassification
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn.functional as F

class BertForSequenceClassificationLSTMGRU(torch.nn.Module):
    def __init__(self, path, config, lstm_hidden_size=512, num_labels=2, lstm_dropout=0.1):
        super().__init__()
        self.num_labels = num_labels
        self.config = config
        self.lstm_hidden_size = lstm_hidden_size
        self.bert = BertModel.from_pretrained(path, config=config)
        self.dropout = nn.Dropout(lstm_dropout)
        # self.pooling = nn.Linear(config.hidden_size, config.hidden_size)
        self.gru = nn.GRU(lstm_hidden_size*2, lstm_hidden_size,
               num_layers=1, bidirectional=True, batch_first=True)
        self.lstm = nn.LSTM(config.hidden_size, lstm_hidden_size,
                          num_layers=1, bidirectional=True, batch_first=True)
        self.classifier = nn.Linear(config.hidden_size*4, num_labels)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None):

        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None

        outputs = self.bert(input_ids=flat_input_ids, position_ids=flat_position_ids,
                            token_type_ids=flat_token_type_ids,
                            attention_mask=flat_attention_mask, head_mask=head_mask)
        bert_output = outputs[0]
        pooled_output = outputs[1]

        h_lstm, _ = self.lstm(bert_output)  # [bs, seq, output*dir]
        h_gru, hh_gru = self.gru(h_lstm)    #
        hh_gru = hh_gru.view(-1, 2 * self.lstm_hidden_size)

        avg_pool = torch.mean(h_gru, 1)
        max_pool, _ = torch.max(h_gru, 1)

        h_conc_a = torch.cat(
            (avg_pool, hh_gru, max_pool, pooled_output), 1
        )


        output = self.dropout(h_conc_a)
        logits = self.classifier(output)

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = [loss, ]
            outputs = outputs + [nn.functional.softmax(logits, -1)]
        else:
            outputs = nn.functional.softmax(logits, -1)
        return outputs  # (loss), logits, (hidden_states), (attentions)
class BertForSequenceClassification_last2embedding_cls(torch.nn.Module):
    def __init__(self,path, config, num_labels=2):
        super().__init__()
        hidden_size_changed = config.hidden_size * 3
        self.num_labels = num_labels
        config.output_hidden_states = True
        self.bert = BertModel.from_pretrained(path, config=config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(hidden_size_changed, num_labels)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None):

        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None

        bert_output_a, pooled_output_a, hidden_output_a = self.bert(input_ids=flat_input_ids, position_ids=flat_position_ids,
                                                                    token_type_ids=flat_token_type_ids, attention_mask=flat_attention_mask, head_mask=head_mask)
        last_cat = torch.cat((pooled_output_a, hidden_output_a[-1][:, 0], hidden_output_a[-2][:, 0]), 1)
        last_cat = self.dropout(last_cat)
        logits = self.classifier(last_cat)

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = [loss, ]
            outputs = outputs + [nn.functional.softmax(logits, -1)]
        else:
            outputs = nn.functional.softmax(logits, -1)

        return outputs  # (loss), logits, (hidden_states), (attentions)
class BertForSequenceClassification_last3embedding(torch.nn.Module):

    def __init__(self, config):
        super(BertForSequenceClassification_last3embedding, self).__init__(config)
        hidden_size_changed = config.hidden_size * 3
        self.num_labels = 2
        config.output_hidden_states = True
        self.bert = BertModel(config)
        self.classifier = nn.Linear(hidden_size_changed, self.num_labels)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None):

        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        bert_output_a, pooled_output_a, hidden_output_a = self.bert(input_ids=flat_input_ids, position_ids=flat_position_ids,
                                token_type_ids=flat_token_type_ids,
                                attention_mask=flat_attention_mask, head_mask=head_mask)

        last_cat = torch.cat(
            (hidden_output_a[-1][:, 0], hidden_output_a[-2][:, 0], hidden_output_a[-3][:, 0]),
            1,
        )

        logits = self.classifier(last_cat)

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = [loss, ]
            outputs = outputs + [nn.functional.softmax(logits, -1)]
        else:
            outputs = nn.functional.softmax(logits, -1)

        return outputs  # (loss), logits, (hidden_states), (attentions)
class BertTextCNNForSequenceClassification(torch.nn.Module):

    def __init__(self,path, config, num_labels=2):
        super().__init__()
        self.filter_sizes = (2, 3, 4)                                   # 鍗风Н鏍稿昂瀵?
        self.num_filters = 256                                          # 鍗风Н鏍告暟閲?channels鏁?
        self.num_labels = 2
        self.bert = BertModel.from_pretrained(path, config=config)
        self.filter_sizes = (2, 3, 4)                                   # 鍗风Н鏍稿昂瀵?
        self.num_filters = 256
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, self.num_filters, (k, config.hidden_size)) for k in self.filter_sizes])

        self.dropout = nn.Dropout(0.5)
        self.fc_cnn = nn.Linear(self.num_filters * len(self.filter_sizes),  self.num_labels)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None):

        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        encoder_out, pooled_output_a = self.bert(input_ids=flat_input_ids, position_ids=flat_position_ids,
                                token_type_ids=flat_token_type_ids,
                                attention_mask=flat_attention_mask, head_mask=head_mask)

        out = encoder_out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        logits = self.fc_cnn(out)

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = [loss, ]
            outputs = outputs + [nn.functional.softmax(logits, -1)]
        else:
            outputs = nn.functional.softmax(logits, -1)

        return outputs  # (loss), logits, (hidden_states), (attentions)
class BertForSequenceClassification(torch.nn.Module):
    def __init__(self,path, config, num_labels=2):
        super().__init__()
        self.num_labels = config.num_labels
        self.bert = BertModel.from_pretrained(path, config=config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None):
        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        outputs = self.bert(input_ids=flat_input_ids, position_ids=flat_position_ids,
                                token_type_ids=flat_token_type_ids,
                                attention_mask=flat_attention_mask, head_mask=head_mask)
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = [loss, ]
            outputs = outputs + [nn.functional.softmax(logits, -1)]
        else:
            outputs = nn.functional.softmax(logits, -1)
        return outputs

class BertForSequenceClassification_last3embedding_cls(torch.nn.Module):
    def __init__(self,path, config, num_labels=2):
        super().__init__()
        hidden_size_changed = config.hidden_size * 4
        self.num_labels = num_labels
        config.output_hidden_states=True
        self.bert = BertModel.from_pretrained(path, config=config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(hidden_size_changed, num_labels)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None):
        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        bert_output_a, pooled_output_a, hidden_output_a = self.bert(input_ids=flat_input_ids, position_ids=flat_position_ids,
                                token_type_ids=flat_token_type_ids,
                                attention_mask=flat_attention_mask, head_mask=head_mask)
        last_cat = torch.cat(
            (pooled_output_a, hidden_output_a[-1][:, 0], hidden_output_a[-2][:, 0], hidden_output_a[-3][:, 0]),
            1,
        )
        pooled_output = self.dropout(last_cat)
        logits = self.classifier(pooled_output)
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = [loss, ]
            outputs = outputs + [nn.functional.softmax(logits, -1)]
        else:
            outputs = nn.functional.softmax(logits, -1)

        return outputs  # (loss), logits, (hidden_states), (attentions)

class BertYnetForSequenceClassification(torch.nn.Module):
    def __init__(self, path, config, num_labels=2):
        super().__init__()
        self.num_labels = config.num_labels
        self.bert1 = BertModel.from_pretrained(path, config=config)
        self.bert2 = BertModel(config=config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size * 2, num_labels)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None):
        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1,
                                                  token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1,
                                                  attention_mask.size(-1)) if attention_mask is not None else None
        outputs1 = self.bert1(input_ids=flat_input_ids, position_ids=flat_position_ids,
                            token_type_ids=flat_token_type_ids,
                            attention_mask=flat_attention_mask, head_mask=head_mask)
        outputs2 = self.bert2(input_ids=flat_input_ids, position_ids=flat_position_ids,
                            token_type_ids=flat_token_type_ids,
                            attention_mask=flat_attention_mask, head_mask=head_mask)
        pooled_output1 = outputs1[1]
        pooled_output2 = outputs2[1]
        last_cat = torch.cat((pooled_output1,pooled_output2),1)

        pooled_output = self.dropout(last_cat)
        logits = self.classifier(pooled_output)

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = [loss, ]
            outputs = outputs + [nn.functional.softmax(logits, -1)]
        else:
            outputs = nn.functional.softmax(logits, -1)
        return outputs
