import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .bert.modeling import *
from collections import OrderedDict
from copy import deepcopy


class MultimodalShiftingGate(nn.Module):
    """ MultimodalShiftingGate """
    def __init__(self, config):
        super(MultimodalShiftingGate, self).__init__()
        # if config.encoding:
        #     self.gv_linear = nn.Linear(config.hidden_size + config.encode_hidden_size, 1)
        #     self.ga_linear = nn.Linear(config.hidden_size + config.encode_hidden_size, 1)
        #     self.weight_v = nn.Parameter(torch.Tensor(config.encode_hidden_size, config.hidden_size))
        #     self.weight_a = nn.Parameter(torch.Tensor(config.encode_hidden_size, config.hidden_size))
        self.gv_linear = nn.Linear(config.hidden_size + config.visual_size, 1)
        self.ga_linear = nn.Linear(config.hidden_size + config.acoustic_size, 1)
        self.weight_v = nn.Parameter(torch.Tensor(config.visual_size, config.hidden_size))
        self.weight_a = nn.Parameter(torch.Tensor(config.acoustic_size, config.hidden_size))
        self.b_h = nn.Parameter(torch.Tensor(config.hidden_size))
        self.beta = config.msg_beta

        self._init_parameters()

    def _init_parameters(self):
        # nn.init.uniform_(self.weight_v)
        # nn.init.uniform_(self.weight_a)
        # nn.init.uniform_(self.b_h)
        bound1 = 1 / math.sqrt(self.weight_v.size(0))
        bound2 = 1 / math.sqrt(self.weight_a.size(0))
        nn.init.uniform_(self.weight_v, -bound1, bound1)
        nn.init.uniform_(self.weight_a, -bound2, bound2)
        nn.init.uniform_(self.b_h, -(bound1+bound2), (bound1+bound2))

    def forward(self, embedding, visual, acoustic):
        n = embedding.size(0)
        gate_v = torch.relu(self.gv_linear(torch.cat([embedding, visual], dim=2)))
        gate_a = torch.relu(self.ga_linear(torch.cat([embedding, acoustic], dim=2)))
        h_v = gate_v * torch.matmul(visual, self.weight_v)
        h_a = gate_a * torch.matmul(acoustic, self.weight_a)
        h_m = h_v + h_a + self.b_h
        alpha = torch.norm(embedding, dim=2) / torch.norm(h_m, dim=2)

        with open('norm_div.json', 'w') as f:
            import json
            json.dump(alpha.squeeze().tolist(), f)
            f.write('\n')

        alpha = self.beta * alpha
        # min of (norm_div, 1.0)
        alpha_mask = (alpha-1) > 0
        alpha = alpha.masked_fill(alpha_mask, 1.0).view(n, -1)
        output = embedding + alpha.unsqueeze(dim=2) * h_m
        return output


class MultimodalBertModel(BertPreTrainedModel):
    """Multimodal BERT model ("Bidirectional Embedding Representations from a Transformer").
    """
    def __init__(self, config, output_attentions=False, keep_multihead_output=False):
        super(MultimodalBertModel, self).__init__(config)
        self.output_attentions = output_attentions
        self.embeddings = BertEmbeddings(config)
        self.shift_gate = MultimodalShiftingGate(config)
        self.encoder = BertEncoder(config, output_attentions=output_attentions,
                                           keep_multihead_output=keep_multihead_output)
        self.pooler = BertPooler(config)
        self.apply(self.init_bert_weights)

    def prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def get_multihead_outputs(self):
        """ Gather all multi-head outputs.
            Return: list (layers) of multihead module outputs with gradients
        """
        return [layer.attention.self.multihead_output for layer in self.encoder.layer]

    def forward(self, input_ids, input_visual, input_acoustic, token_type_ids=None, attention_mask=None,
                output_all_encoded_layers=True, head_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand_as(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype) # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(input_ids, token_type_ids)
        shift_output = self.shift_gate(embedding_output, input_visual, input_acoustic)
        encoded_layers = self.encoder(shift_output,
                                      extended_attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers,
                                      head_mask=head_mask)
        if self.output_attentions:
            all_attentions, encoded_layers = encoded_layers
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        if self.output_attentions:
            return all_attentions, encoded_layers, pooled_output
        return encoded_layers, pooled_output


class MultimodalBertForClassification(BertPreTrainedModel):
    """ Multimodal-BERT for classification"""
    def __init__(self, config, 
                 visual_size=47,
                 acoustic_size=74,
                 msg_beta=0.0,
                 num_labels=2, 
                 output_attentions=False, 
                 keep_multihead_output=False):
        config.visual_size = visual_size
        config.acoustic_size = acoustic_size
        config.msg_beta = msg_beta

        super(MultimodalBertForClassification, self).__init__(config)
        self.output_attentions = output_attentions
        self.num_labels = num_labels
        self.bert = MultimodalBertModel(config, output_attentions=output_attentions,
                                        keep_multihead_output=keep_multihead_output)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, input_visual, input_acoustic, token_type_ids=None, attention_mask=None, labels=None, head_mask=None):
        outputs = self.bert(input_ids, input_visual, input_acoustic, token_type_ids, attention_mask,
                            output_all_encoded_layers=False, head_mask=head_mask)
        if self.output_attentions:
            all_attentions, _, pooled_output = outputs
        else:
            _, pooled_output = outputs
        pooled_output = self.dropout(pooled_output)

        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = nn.MSELoss()
            loss = loss_fct(logits.view(-1), labels.view(-1))
            return loss
        elif self.output_attentions:
            return all_attentions, logits
        return logits

