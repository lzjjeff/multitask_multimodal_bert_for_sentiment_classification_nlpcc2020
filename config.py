import json
import copy


class Config(object):
    """
        dataset: mosi | mosei
    """
    def __init__(self):
        self.task_type = "sentiment classification"
        self.dataset = "mosi"

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path):
        """ Save this instance to a json file."""
        with open(json_file_path, "w", encoding='utf-8') as writer:
            writer.write(self.to_json_string())


class GlobalConfig(Config):
    def __init__(self):
        """
            task_type: a subset of ["multimodal sentiment classification",
                                    "visual sentiment classification",
                                    "acoustic sentiment classification",
                                    ]
            plot_types: a subset of ["multi", "visual", "acoustic"]
        """
        super(GlobalConfig, self).__init__()
        self.task_type = ["multimodal sentiment classification", "visual sentiment classification",
                          "acoustic sentiment classification"
                          ]
        self.output_dir = ""
        self.cache_dir = ""

        self.do_train = True
        self.do_eval = True
        self.do_predict = True
        self.mix_opt = 0
        self.ratio = 0
        self.num_train_epochs = 6.0
        self.no_cuda = False
        self.overwrite_output_dir = False
        self.device_ids = [0]
        self.seed = 123456
        self.gradient_accumulation_steps = 1
        self.plot_types = ["multi", "visual", "acoustic"]
        # for distributed training
        self.local_rank = -1
        # for fp16
        self.fp16 = False
        self.loss_scale = 0


class MBertClfConfig(Config):
    def __init__(self):
        """
            bert_config: bert-base-uncased | bert-large-uncased-whole-word-masking
        """
        super(MBertClfConfig, self).__init__()
        self.is_train = True
        self.task_type = "multimodal sentiment classification"
        self.data_dir = "./data/mosi/" if self.dataset == "mosi" else "./data/mosei/"
        self.bert_config = "bert-base-uncased"
        self.task_name = "mosi" if self.dataset == "mosi" else "mosei"
        self.msg_beta = 0.0
        self.num_labels = 1
        self.max_seq_length = 128
        self.pad_mode = "zero"
        self.do_lower_case = True
        self.output_attentions = False
        self.keep_multihead_output = False
        self.train_batch_size = 4
        self.eval_batch_size = 8
        self.predict_batch_size = 16
        self.lr = 5e-6
        self.weight_decay = 0.01
        self.warmup_proportion = 0.1


class MultiTaskConfig(Config):
    def __init__(self):
        super(MultiTaskConfig, self).__init__()
        self.do_vsc = True
        self.do_asc = True
        self.msl = "linear"
        self.encoder = "rnn-attn-cat"


class TransformerConfig(Config):
    def __init__(self):
        super(TransformerConfig, self).__init__()
        self.attention_probs_dropout_prob = 0.1
        self.hidden_act = "gelu"
        self.hidden_dropout_prob = 0.1
        self.hidden_size = 768
        # self.visual_size = 47
        # self.acoustic_size = 74
        self.initializer_range = 0.02
        self.intermediate_size = 1024
        self.num_attention_heads = 12
        self.num_hidden_layers = 12
        self.type_vocab_size = 2
        self.layer_norm_eps = 1e-12


class VisualClfConfig(Config):
    def __init__(self):
        super(VisualClfConfig, self).__init__()
        self.is_train = True
        self.task_type = "visual sentiment classification"
        self.data_folder = ''
        self.data_name = ''
        self.num_labels = 1

        self.input_dim = 1280
        self.mapped_dim = 47 if self.dataset == "mosi" else 35
        self.rnn_hidden_dim = 128
        self.num_rnn_layers = 1
        self.fc_dim = 128
        self.num_fc_layers = 3
        self.dropout = 0.1

        self.train_batch_size = 32
        self.lr = 1e-3
        self.weight_decay = 1e-4
        self.learning_anneal = 1.01
        

class AcousticClfConfig(Config):
    def __init__(self):
        super(AcousticClfConfig, self).__init__()
        self.is_train = True
        self.task_type = "acoustic sentiment classification"
        self.data_folder = ''
        self.data_name = ''
        self.num_labels = 1

        self.input_dim = 1312
        self.mapped_dim = 74
        self.rnn_hidden_dim = 128
        self.num_rnn_layers = 1
        self.fc_dim = 128
        self.num_fc_layers = 3
        self.dropout = 0.1

        self.train_batch_size = 32
        self.lr = 1e-3
        self.weight_decay = 1e-4
        self.learning_anneal = 1.01



