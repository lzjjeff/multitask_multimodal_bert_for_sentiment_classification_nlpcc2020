import os
import sys
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


sys.path.append("../")
from model.mbert import MultimodalBertForClassification as MultiModalClassifier
from model.bert.modeling import BertPreTrainedModel


class MultiModalMapping(nn.Module):
    def __init__(self, visual_size, acoustic_size, active=False):
        super(MultiModalMapping, self).__init__()
        self.active = active

        self.visual_fc = nn.Linear(visual_size, visual_size)
        self.acoustic_fc = nn.Linear(acoustic_size, acoustic_size)
        if active:
            self.visual_act = nn.ReLU()
            self.acoustic_act = nn.ReLU()

    def visual_mapping(self, input_visual):
        if self.active:
            return self.visual_act(self.visual_fc(input_visual))
        return self.visual_fc(input_visual)

    def acoustic_mapping(self, input_acoustic):
        if self.active:
            return self.acoustic_act(self.acoustic_fc(input_acoustic))
        return self.acoustic_fc(input_acoustic)


class MultiModalMultiTaskSentimentClassification(nn.Module):
    def __init__(self, 
                 mbertclf_config,
                 visualclf_config,
                 acousticclf_config,
                 multitask_config):
        super(MultiModalMultiTaskSentimentClassification, self).__init__()

        self.config1 = mbertclf_config
        self.config2 = visualclf_config
        self.config3 = acousticclf_config
        self.config0 = multitask_config

        self.mbert_clf = MultiModalClassifier.from_pretrained(pretrained_model_name_or_path=self.config1.bert_config,
                                                              visual_size=self.config2.mapped_dim,
                                                              acoustic_size=self.config3.mapped_dim,
                                                              msg_beta=self.config1.msg_beta,
                                                              num_labels=self.config1.num_labels,
                                                              output_attentions=self.config1.output_attentions,
                                                              keep_multihead_output=self.config1.keep_multihead_output)

        if self.config0.msl == "linear":
            self.msl_linear = MultiModalMapping(visual_size=self.config2.mapped_dim,
                                                acoustic_size=self.config3.mapped_dim,
                                                active=True)

        if self.config0.encoder == "rnn-attn-add":
            m = 2
        elif self.config0.encoder == "rnn-attn-cat":
            m = 4

        # RNN
        self.visual_rnn = nn.LSTM(input_size=self.config2.mapped_dim,
                                  hidden_size=self.config2.rnn_hidden_dim,
                                  num_layers=self.config2.num_rnn_layers,
                                  dropout=self.config2.dropout,
                                  batch_first=True,
                                  bidirectional=True)
        self.acoustic_rnn = nn.LSTM(input_size=self.config3.mapped_dim,
                                    hidden_size=self.config3.rnn_hidden_dim,
                                    num_layers=self.config3.num_rnn_layers,
                                    dropout=self.config3.dropout,
                                    batch_first=True,
                                    bidirectional=True)
        # Linear Attention
        self.visual_attn = nn.Linear(self.config2.rnn_hidden_dim*2, 1, bias=False)
        self.acoustic_attn = nn.Linear(self.config3.rnn_hidden_dim*2, 1, bias=False)

        if "rnn" in self.config0.msl:
            if self.config0.msl == "rnn":
                self.visual_proj = nn.Linear(self.config2.fc_dim * 2, self.config2.mapped_dim)
                self.acoustic_proj = nn.Linear(self.config3.fc_dim * 2, self.config3.mapped_dim)
            else:
                self.visual_proj = nn.Linear(self.config2.fc_dim * m, self.config2.mapped_dim)
                self.acoustic_proj = nn.Linear(self.config3.fc_dim * m, self.config3.mapped_dim)

        self.visual_clf = nn.Sequential(
            nn.Dropout(self.config2.dropout),
            nn.Linear(self.config2.fc_dim*m, self.config2.fc_dim*2),
            nn.ReLU(),
            nn.Linear(self.config2.fc_dim*2, self.config2.fc_dim),
            nn.ReLU(),
            nn.Linear(self.config2.fc_dim, self.config2.num_labels)
        )

        self.acoustic_clf = nn.Sequential(
            nn.Dropout(self.config3.dropout),
            nn.Linear(self.config3.fc_dim*m, self.config3.fc_dim*2),
            nn.ReLU(),
            nn.Linear(self.config3.fc_dim*2, self.config3.fc_dim),
            nn.ReLU(),
            nn.Linear(self.config3.fc_dim, self.config3.num_labels)
        )

    def save_config_to_json_file(self, output_dir):
        self.config0.to_json_file(os.path.join(output_dir, "config_multitask.json"))
        self.config1.to_json_file(os.path.join(output_dir, "config_mbert_clf.json"))
        self.config2.to_json_file(os.path.join(output_dir, "config_visual_clf.json"))
        self.config3.to_json_file(os.path.join(output_dir, "config_acoustic_clf.json"))

    def rnn_step(self, input, lengths, rnn):
        packed = pack_padded_sequence(input, lengths.cpu(), batch_first=True)
        rnn.flatten_parameters()
        packed_h, (final_h, _) = rnn(packed)
        padded_h, _ = pad_packed_sequence(packed_h, batch_first=True)
        final_h = torch.cat([final_h[0], final_h[1]], dim=1)
        return padded_h, final_h

    def forward(self, batch):
        # print(batch)
        # print(len(batch["data"][-1].shape))
        if len(batch["data"][-1].shape) == 0:
            batch["data"] = [t.unsqueeze(0) for t in batch["data"]]
        # print(batch)
        # print(len(batch["data"][-1].shape))
        if batch["data_type"][0] == "multi":
            input_ids, input_visual, input_acoustic, input_lengths, input_mask, segment_ids, label_ids = batch["data"]
            batch_size = input_ids.size(0)
            max_len = input_ids.size(1)

            if self.config0.msl == "linear":
                input_visual = self.msl_linear.visual_mapping(input_visual)
                input_acoustic = self.msl_linear.acoustic_mapping(input_acoustic)
            elif "rnn" in self.config0.msl:
                input_lengths, idx = input_lengths.sort(dim=-1, descending=True)
                input_visual = input_visual[idx]
                input_acoustic = input_acoustic[idx]
                padded_hv, final_hv = self.rnn_step(input_visual, input_lengths, self.visual_rnn)
                padded_ha, final_ha = self.rnn_step(input_acoustic, input_lengths, self.acoustic_rnn)
                if self.config0.msl == "rnn-attn":
                    alpha_v = torch.softmax(self.visual_attn(padded_hv), dim=1).permute(0, 2, 1)
                    alpha_a = torch.softmax(self.acoustic_attn(padded_ha), dim=1).permute(0, 2, 1)
                    attn_hv = torch.matmul(alpha_v, padded_hv).view(batch_size, -1)
                    attn_ha = torch.matmul(alpha_a, padded_ha).view(batch_size, -1)
                    if self.config0.encoder == "rnn-attn-cat":
                        final_hv = torch.cat([attn_hv, final_hv], dim=1)
                        final_ha = torch.cat([attn_ha, final_ha], dim=1)
                    elif self.config0.encoder == "rnn-attn-add":
                        final_hv = final_hv + attn_hv
                        final_ha = final_ha + attn_ha
                input_visual = self.visual_proj(final_hv).unsqueeze(1).expand(batch_size, max_len, self.config2.mapped_dim)
                input_acoustic = self.acoustic_proj(final_ha).unsqueeze(1).expand(batch_size, max_len, self.config3.mapped_dim)

            logits = self.mbert_clf(input_ids, input_visual, input_acoustic,
                                    token_type_ids=segment_ids,
                                    attention_mask=input_mask)
            return logits.view(-1), label_ids.view(-1)

        elif batch["data_type"][0] == "visual" and self.config0.do_vsc:
            input_visual, lengths, labels = batch["data"]
            batch_size = input_visual.size(0)
            max_len = input_visual.size(1)

            if self.config0.msl == "linear":
                input_visual = self.msl_linear.visual_mapping(input_visual)

            # rnn-attn-res step
            lengths, idx = lengths.sort(dim=-1, descending=True)
            input_visual = input_visual[idx]
            labels = labels[idx]
            padded_h, final_h = self.rnn_step(input_visual, lengths, self.visual_rnn)
            alpha = torch.softmax(self.visual_attn(padded_h), dim=1).permute(0, 2, 1)
            attn_h = torch.matmul(alpha, padded_h).view(batch_size, -1)
            if self.config0.encoder == "rnn-attn-cat":
                final_h = torch.cat([attn_h, final_h], dim=1)
            elif self.config0.encoder == "rnn-attn-add":
                final_h = final_h + attn_h
            logits = self.visual_clf(final_h)

            return logits.view(-1),  labels.view(-1)

        elif batch["data_type"][0] == "acoustic" and self.config0.do_asc:
            input_acoustic, lengths, labels = batch["data"]
            batch_size = input_acoustic.size(0)
            max_len = input_acoustic.size(1)

            if self.config0.msl == "linear":
                input_acoustic = self.msl_linear.acoustic_mapping(input_acoustic)

            # rnn-attn-res step
            lengths, idx = lengths.sort(dim=-1, descending=True)
            input_acoustic = input_acoustic[idx]
            labels = labels[idx]
            padded_h, final_h = self.rnn_step(input_acoustic, lengths, self.acoustic_rnn)
            alpha = torch.softmax(self.acoustic_attn(padded_h), dim=1).permute(0, 2, 1)
            attn_h = torch.matmul(alpha, padded_h).view(batch_size, -1)
            if self.config0.encoder == "rnn-attn-cat":
                final_h = torch.cat([attn_h, final_h], dim=1)
            elif self.config0.encoder == "rnn-attn-add":
                final_h = final_h + attn_h
            logits = self.acoustic_clf(final_h)

            return logits.view(-1), labels.view(-1)

        else:
            # raise ValueError("Wrong data type: %s" % batch["data_type"])
            return None
