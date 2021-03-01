from torch.nn.functional import gelu, elu
import torch.nn.functional as F
import torch.nn as nn
import math
import torch
import os
from transformers import WEIGHTS_NAME, AutoModel
from transformers.modeling_roberta import create_position_ids_from_input_ids
from copy import deepcopy
from meta_neural_network_architectures import *


def per_step_layer_norm(model, num_steps):
    for child_name, child in model.named_children():
        if isinstance(child, MetaLayerNormLayer):
            child.init_per_step_weights(num_steps)
        else:
            per_step_layer_norm(child, num_steps)


class MetaBERT(nn.Module):
    def __init__(
        self, config, is_distil, is_xlm, per_step_layer_norm_weights=True, device="cpu"
    ):
        super(MetaBERT, self).__init__()  # config)

        self.is_distil = is_distil
        self.is_xlm = is_xlm
        self.per_step_layer_norm_weights = per_step_layer_norm_weights

        self.config = config
        if is_xlm:
            self.embeddings = MetaRoBertaEmbedding(config, is_distil)
        else:
            self.embeddings = MetaBertEmbedding(config, is_distil)
        self.encoder = MetaBertEncoder(config, is_distil)

        self.classifier = MetaBertClassHead(config)

        self.fast_weights = None
        self.device = torch.device(device)

    def freeze(self, freeze_classifier=False):
        # Freeze the model up to the classification head
        for p in self.embeddings.parameters():
            p.requires_grad = False

        for p in self.encoder.parameters():
            p.requires_grad = False

        if freeze_classifier:
            for p in self.classifier.parameters():
                p.requires_grad = False

    def unfreeze(self):

        for p in self.parameters():
            p.requires_grad = True

    def get_inner_loop_params(self):

        params = {
            param_name: param.to(self.device)
            for param_name, param in self.named_parameters()
        }
        return params

    @classmethod
    def init_from_pretrained(
        cls,
        state_dict,
        config,
        num_labels,
        is_distil,
        is_xlm,
        per_step_layer_norm_weights=True,
        num_inner_loop_steps=None,
        device="cpu",
    ):

        config.num_labels = num_labels

        if is_xlm:
            state_dict = {
                k.replace("pooler", "classifier").replace("roberta.", "", 1): v
                for k, v in state_dict.items()
            }

        if is_distil:  # convert differences in naming
            state_dict = {
                distil_state_dict_to_bert(k): v for k, v in state_dict.items()
            }
            config = distil_to_bert_config(config)

        self = cls(config, is_distil, is_xlm, device=device)

        curr_state_dict = self.state_dict()
        for k in curr_state_dict.keys():
            if k not in state_dict.keys():
                print(
                    "Warning! Parameter {} not loaded from pre-trained checkpoint.".format(
                        k
                    )
                )

        # Load the weights
        self.load_state_dict(state_dict, strict=False)

        if per_step_layer_norm_weights:  # init per step norm layer weights
            per_step_layer_norm(self, num_inner_loop_steps)

        return self

    def forward(
        self,
        num_step,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        params=None,
        training=None,
        return_hidden_states=False,
        return_pooled=False,
    ):
        embedding_params = None
        encoder_params = None
        class_head_params = None
        if params is not None:
            params = extract_top_level_dict(current_dict=params)
            embedding_params = params["embeddings"]
            encoder_params = params["encoder"]
            class_head_params = params["classifier"]

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, input_shape, self.device
        )

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        # if self.config.is_decoder and encoder_hidden_states is not None:
        #     encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
        #     encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
        #     if encoder_attention_mask is None:
        #         encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
        #     encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        # else:
        encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            params=embedding_params,
        )

        encoder_outputs = self.encoder(
            hidden_states=embedding_output,
            num_step=num_step,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            params=encoder_params,
        )
        sequence_output = encoder_outputs[0]

        logits_or_pooled = self.classifier(
            sequence_output, return_pooled=return_pooled, params=class_head_params
        )
        if return_hidden_states:
            return logits_or_pooled, sequence_output
        else:
            return (logits_or_pooled,)

    def get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
        """
        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        attention_probs has shape bsz x n_heads x N x N
        Arguments:
            head_mask: torch.Tensor or None: has shape [num_heads] or [num_hidden_layers x num_heads]
            num_hidden_layers: int
        Returns:
             Tensor of shape shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
             or list with [None] for each layer
        """
        if head_mask is not None:
            head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
            if is_attention_chunked is True:
                head_mask = head_mask.unsqueeze(-1)
        else:
            head_mask = [None] * num_hidden_layers

        return head_mask

    def get_extended_attention_mask(self, attention_mask, input_shape, device):
        """Makes broadcastable attention mask and causal mask so that future and maked tokens are ignored.
        Arguments:
            attention_mask: torch.Tensor with 1 indicating tokens to ATTEND to
            input_shape: tuple, shape of input_ids
            device: torch.Device, usually self.device
        Returns:
            torch.Tensor with dtype of attention_mask.dtype
        """
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if self.config.is_decoder:
                batch_size, seq_length = input_shape
                seq_ids = torch.arange(seq_length, device=device)
                causal_mask = (
                    seq_ids[None, None, :].repeat(batch_size, seq_length, 1)
                    <= seq_ids[None, :, None]
                )
                # causal and attention masks must have same type with pytorch version < 1.3
                causal_mask = causal_mask.to(attention_mask.dtype)
                extended_attention_mask = (
                    causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
                )
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
                    input_shape, attention_mask.shape
                )
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        # extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask


####################################################
# HELPER FUNCTIONS
####################################################


def distil_to_bert_config(config):
    config.layer_norm_eps = 1e-12
    config.intermediate_size = config.hidden_dim
    config.hidden_act = config.activation

    return config


def distil_state_dict_to_bert(k):

    k = k.replace("distilbert.", "")
    k = k.replace("preclassifier.", "classifier.dense.")
    k = k.replace("classifier.weight", "classifier.out_proj.weight")
    k = k.replace("classifier.bias", "classifier.out_proj.bias")
    k = k.replace("transformer", "encoder")
    k = k.replace("q_lin", "self.query")
    k = k.replace("k_lin", "self.key")
    k = k.replace("v_lin", "self.value")
    k = k.replace("out_lin", "output.dense")
    k = k.replace("sa_layer_norm", "attention.output.LayerNorm")
    k = k.replace("output_layer_norm", "output.LayerNorm")
    k = k.replace("ffn.lin1", "intermediate.dense")
    k = k.replace("ffn.lin2", "output.dense")

    return k


if __name__ == "__main__":

    is_distil = False
    is_xlm = True
    bert = AutoModel.from_pretrained("xlm-roberta-base")
    bert.eval()
    t = bert.state_dict()
    config = bert.config
    classifier = MetaBERT.init_from_pretrained(
        t,
        config,
        num_labels=4,
        is_distil=is_distil,
        is_xlm=is_xlm,
        per_step_layer_norm_weights=True,
        num_inner_loop_steps=5,
        init_class_head=True,
    )
    classifier.eval()
    s = classifier.state_dict()
    # meta_bert.load_state_dict(t)

    # model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
    # fast_weights = OrderedDict(model.named_parameters())
    #
    input_ids = torch.Tensor(
        [
            [101, 1303, 1110, 1199, 3087, 1106, 4035, 13775, 102],
            [101, 178, 1274, 1204, 1176, 1115, 4170, 182, 102],
        ]
    ).to(torch.long)

    m_out = classifier(0, input_ids=input_ids)
    b_out = bert(input_ids)

    for (n1, p1), (n2, p2) in zip(
        classifier.named_parameters(), bert.named_parameters()
    ):
        if p1.data.ne(p2.data).sum() > 0:
            print(n1, n2, "False")

    assert (
        m_out.ne(b_out[0])
    ).sum() == 0, "Output not consistent between MetaBert and Bert"
