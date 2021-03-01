import numbers
from copy import copy

import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import math
from transformers.modeling_utils import prune_linear_layer
from transformers.modeling_roberta import create_position_ids_from_input_ids
from transformers.modeling_bert import ACT2FN

META_ADAPTER_NAME = "adapter"


def extract_top_level_dict(current_dict):
    """
    Builds a graph dictionary from the passed depth_keys, value pair. Useful for dynamically passing external params
    :param depth_keys: A list of strings making up the name of a variable. Used to make a graph for that params tree.
    :param value: Param value
    :param key_exists: If none then assume new dict, else load existing dict and add new key->value pairs to it.
    :return: A dictionary graph of the params already added to the graph.
    """
    output_dict = dict()
    for key in current_dict.keys():
        name = key.replace("layer_dict.", "")
        top_level = name.split(".")[0]
        sub_level = ".".join(name.split(".")[1:])

        if top_level not in output_dict:
            if sub_level == "":
                output_dict[top_level] = current_dict[key]
            else:
                output_dict[top_level] = {sub_level: current_dict[key]}
        else:
            new_item = {key: value for key, value in output_dict[top_level].items()}
            new_item[sub_level] = current_dict[key]
            output_dict[top_level] = new_item

    # print(current_dict.keys(), output_dict.keys())
    return output_dict


class Activation_Function_Class(nn.Module):
    """
    Implementation of various activation function. For Adapter block
    """

    def __init__(self, hidden_act):

        if hidden_act.lower() == "relu":
            self.f = nn.functional.relu
        elif hidden_act.lower() == "tanh":
            self.f = torch.tanh
        elif hidden_act.lower() == "swish":

            def swish(x):
                return x * torch.sigmoid(x)

            self.f = swish
        elif hidden_act.lower() == "gelu":

            def gelu_new(x):
                """
                Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
                Also see https://arxiv.org/abs/1606.08415
                """
                return (
                    0.5
                    * x
                    * (
                        1
                        + torch.tanh(
                            math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))
                        )
                    )
                )

            self.f = gelu_new
        elif hidden_act.lower() == "leakyrelu":
            self.f = nn.functional.leaky_relu

        super().__init__()

    def forward(self, x):
        return self.f(x)


class MetaEmbedding(nn.Module):
    def __init__(self, num_rows, dim, padding_idx=None):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(num_rows, dim))
        self.padding_idx = padding_idx

    def forward(self, x, params=None):

        weight = None
        if params is not None:
            params = extract_top_level_dict(params)
            weight = params["weight"]

        if weight is None:
            weight = self.weight

        return F.embedding(input=x, weight=weight, padding_idx=self.padding_idx)


class MetaBertEmbedding(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config, is_distil):
        super().__init__()
        # Save embedding parameter in module instead of embedding itself
        self.word_embeddings = MetaEmbedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.position_embeddings = MetaEmbedding(
            config.max_position_embeddings, config.hidden_size
        )

        # is_distil = False #TODO: temp
        if not is_distil:
            self.token_type_embeddings = MetaEmbedding(
                config.type_vocab_size, config.hidden_size
            )

        self.is_distil = is_distil

        self.hidden_size = config.hidden_size
        self.layer_norm_eps = config.layer_norm_eps
        self.dropout_prob = config.dropout if is_distil else config.hidden_dropout_prob

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = MetaLayerNormLayer(config.hidden_size, eps=self.layer_norm_eps)
        self.dropout_prob = config.dropout if is_distil else config.hidden_dropout_prob

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        params=None,
        num_step=0,
        is_train=False,
    ):

        word_embedding_params = None
        position_embeddings_params = None
        token_type_embeddings_params = None
        layer_norm_params = None

        if params is not None:
            params = extract_top_level_dict(current_dict=params)
            word_embedding_params = params["word_embeddings"]
            position_embeddings_params = params["position_embeddings"]
            if not self.is_distil:
                token_type_embeddings_params = params["token_type_embeddings"]

            layer_norm_params = params.get("LayerNorm", None)

        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            # if self.is_xlm:
            #     position_ids += 2
            position_ids = position_ids.unsqueeze(0).expand(input_shape)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(
                input_ids, params=word_embedding_params
            )

        position_embeddings = self.position_embeddings(
            position_ids, params=position_embeddings_params
        )

        embeddings = inputs_embeds + position_embeddings
        if not self.is_distil:
            token_type_embeddings = self.token_type_embeddings(
                token_type_ids, params=token_type_embeddings_params
            )
            embeddings += token_type_embeddings

        embeddings = self.LayerNorm(
            embeddings, params=layer_norm_params, num_step=num_step
        )

        if self.training:
            embeddings = F.dropout(embeddings, p=self.dropout_prob)

        return embeddings


class MetaRoBertaEmbedding(MetaBertEmbedding):
    def __init__(self, config, is_distil=False):
        super().__init__(config, is_distil)
        self.padding_idx = config.pad_token_id
        self.word_embeddings = MetaEmbedding(
            config.vocab_size, config.hidden_size, padding_idx=self.padding_idx
        )
        self.position_embeddings = MetaEmbedding(
            config.max_position_embeddings,
            config.hidden_size,
            padding_idx=self.padding_idx,
        )

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        params=None,
    ):

        if position_ids is None:
            if input_ids is not None:
                # Create the position ids from the input token ids. Any padded tokens remain padded.
                position_ids = create_position_ids_from_input_ids(
                    input_ids, self.padding_idx
                ).to(input_ids.device)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(
                    inputs_embeds
                )

        return super().forward(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            params=params,
        )

    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """We are provided embeddings directly. We cannot infer which are padded so just generate
        sequential position ids.

        :param torch.Tensor inputs_embeds:
        :return torch.Tensor:
        """
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        position_ids = torch.arange(
            self.padding_idx + 1,
            sequence_length + self.padding_idx + 1,
            dtype=torch.long,
            device=inputs_embeds.device,
        )
        return position_ids.unsqueeze(0).expand(input_shape)


class MetaBertSelfAttention(nn.Module):
    def __init__(self, config, is_distil):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(
            config, "embedding_size"
        ):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )
        self.output_attentions = config.output_attentions

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = MetaLinearLayer(config.hidden_size, self.all_head_size)
        self.key = MetaLinearLayer(config.hidden_size, self.all_head_size)
        self.value = MetaLinearLayer(config.hidden_size, self.all_head_size)

        self.dropout_prob = (
            config.attention_dropout
            if is_distil
            else config.attention_probs_dropout_prob
        )

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        num_step,
        params=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        key_params = None
        value_params = None
        query_params = None
        if params is not None:
            params = extract_top_level_dict(current_dict=params)
            key_params = params["key"]
            value_params = params["value"]
            query_params = params["query"]
        mixed_query_layer = self.query(hidden_states, params=query_params)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states, params=key_params)
            mixed_value_layer = self.value(encoder_hidden_states, params=value_params)
            attention_mask = encoder_attention_mask
        else:
            mixed_key_layer = self.key(hidden_states, params=key_params)
            mixed_value_layer = self.value(hidden_states, params=value_params)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        if self.training:
            attention_probs = F.dropout(attention_probs, p=self.dropout_prob)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (
            (context_layer, attention_probs)
            if self.output_attentions
            else (context_layer,)
        )
        return outputs


class MetaBertSelfOutput(nn.Module):
    def __init__(self, config, is_distil):
        super().__init__()
        self.dense = MetaLinearLayer(config.hidden_size, config.hidden_size)
        self.layer_norm_eps = 1e-12 if is_distil else config.layer_norm_eps
        self.LayerNorm = MetaLayerNormLayer(config.hidden_size, eps=self.layer_norm_eps)
        self.dropout_prob = config.hidden_dropout_prob

    def forward(self, hidden_states, input_tensor, num_step, params=None):

        dense_params = None
        layer_norm_params = None
        if params is not None:
            params = extract_top_level_dict(current_dict=params)
            dense_params = params["dense"]
            layer_norm_params = params.get("LayerNorm", None)

        hidden_states = self.dense(hidden_states, params=dense_params)
        if self.training:
            hidden_states = F.dropout(hidden_states, self.dropout_prob)
        hidden_states = self.LayerNorm(
            hidden_states + input_tensor, num_step=num_step, params=layer_norm_params
        )

        return hidden_states


class MetaAdapterBertSelfOutput(nn.Module):
    def __init__(self, config, is_distil):
        super().__init__()
        self.dense = MetaLinearLayer(config.hidden_size, config.hidden_size)
        self.layer_norm_eps = 1e-12 if is_distil else config.layer_norm_eps
        self.LayerNorm = MetaLayerNormLayer(config.hidden_size, eps=self.layer_norm_eps)
        self.dropout_prob = config.hidden_dropout_prob

        self.attention_text_task_adapters = nn.ModuleDict(
            {
                META_ADAPTER_NAME: MetaAdapterLayer(
                    input_size=config.hidden_size,
                    down_sample=config.adapter_down_sample_size,
                    non_linearity=config.adapters.config_map[
                        "text_task"
                    ].non_linearity.lower(),
                )
            }
        )
        self.adapter_fusion_layer = nn.ModuleDict(dict())
        self.attention_text_lang_adapters = nn.ModuleDict(dict())

    def forward(self, hidden_states, input_tensor, num_step, params=None):

        dense_params = None
        layer_norm_params = None
        adapter_params = None
        if params is not None:
            params = extract_top_level_dict(current_dict=params)
            dense_params = params["dense"]
            layer_norm_params = params.get("LayerNorm", None)

            adapter_params = extract_top_level_dict(
                params["attention_text_task_adapters"]
            )[META_ADAPTER_NAME]

        hidden_states = self.dense(hidden_states, params=dense_params)
        if self.training:
            hidden_states = F.dropout(hidden_states, self.dropout_prob)
        hidden_states = self.LayerNorm(
            hidden_states, num_step=num_step, params=layer_norm_params
        )

        adapter = self.attention_text_task_adapters[META_ADAPTER_NAME]
        hidden_states = adapter(hidden_states, params=adapter_params)

        return hidden_states + input_tensor


class MetaBertAttention(nn.Module):
    def __init__(self, config, is_distil, use_adapter=False):
        super().__init__()
        self.self = MetaBertSelfAttention(config, is_distil)
        if use_adapter:
            self.output = MetaAdapterBertSelfOutput(config, is_distil)
        else:
            self.output = MetaBertSelfOutput(config, is_distil)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        mask = torch.ones(self.self.num_attention_heads, self.self.attention_head_size)
        heads = (
            set(heads) - self.pruned_heads
        )  # Convert to set and remove already pruned heads
        for head in heads:
            # Compute how many pruned heads are before the head and move the index accordingly
            head = head - sum(1 if h < head else 0 for h in self.pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = (
            self.self.attention_head_size * self.self.num_attention_heads
        )
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states,
        num_step,
        params=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        self_params = None
        output_params = None
        if params is not None:
            params = extract_top_level_dict(current_dict=params)
            self_params = params["self"]
            output_params = params["output"]

        self_outputs = self.self(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            params=self_params,
            num_step=num_step,
        )
        attention_output = self.output(
            self_outputs[0], hidden_states, params=output_params, num_step=num_step
        )
        outputs = (attention_output,) + self_outputs[
            1:
        ]  # add attentions if we output them
        return outputs


class MetaBertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = MetaLinearLayer(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states, params=None):

        dense_params = None
        if params is not None:
            params = extract_top_level_dict(current_dict=params)
            dense_params = params["dense"]

        hidden_states = self.dense(hidden_states, params=dense_params)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class MetaBertOutput(nn.Module):
    def __init__(self, config, is_distil):
        super().__init__()
        self.dense = MetaLinearLayer(config.intermediate_size, config.hidden_size)
        self.layer_norm_eps = 1e-12 if is_distil else config.layer_norm_eps
        self.LayerNorm = MetaLayerNormLayer(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.dropout_prob = config.hidden_dropout_prob

    def forward(self, hidden_states, input_tensor, num_step, params=None):

        dense_params = None
        layer_norm_params = None
        if params is not None:
            params = extract_top_level_dict(current_dict=params)
            dense_params = params["dense"]
            layer_norm_params = params.get("LayerNorm", None)

        hidden_states = self.dense(hidden_states, params=dense_params)
        if self.training:
            hidden_states = F.dropout(hidden_states, p=self.dropout_prob)

        hidden_states = self.LayerNorm(
            hidden_states + input_tensor, num_step=num_step, params=layer_norm_params
        )
        return hidden_states


# class MetaSequential(nn.Sequential):
#
#     def __init__(self, *args):
#         super(MetaSequential, self).__init__(*args)
#
#     def forward(self, input, params=None):
#
#         if params is None:
#             for module in self:
#                 input = module(input)
#         else:
#             for module, param in zip(self, params):
#                 input = module(input, params=param)
#
#         return input


class MetaAdapterLayer(nn.Module):
    """
    Implementation of a single Adapter block.
    """

    def __init__(
        self,
        input_size,
        down_sample=None,
        non_linearity="relu",
        init_bert_weights=True,
        add_layer_norm_before=False,
        add_layer_norm_after=False,
        residual_before_ln=True,
    ):
        super().__init__()

        self.input_size = input_size
        self.add_layer_norm_before = add_layer_norm_before
        self.add_layer_norm_after = add_layer_norm_after
        self.residual_before_ln = residual_before_ln

        # list for all modules of the adapter, passed into nn.Sequential()
        seq_list = []

        # If we want to have a layer norm on input, we add it to seq_list
        # if self.add_layer_norm_before:
        #     self.adapter_norm_before = MetaLayerNormLayer(self.input_size)
        #     seq_list.append(self.adapter_norm_before)

        # if a downsample size is not passed, we just half the size of the original input
        self.down_sample = down_sample
        if down_sample is None:
            self.down_sample = self.input_size // 2

        # Linear down projection of the input
        seq_list.append(MetaLinearLayer(self.input_size, self.down_sample))

        # select non-linearity
        self.non_linearity = Activation_Function_Class(non_linearity.lower())

        seq_list.append(self.non_linearity)

        # sequential adapter, first downproject, then non-linearity then upsample. In the forward pass we include the
        # residual connection
        self.adapter_down = nn.Sequential(*seq_list)

        # Up projection to input size
        self.adapter_up = MetaLinearLayer(self.down_sample, self.input_size)

        # If we want to have a layer norm on output, we apply it later after a separate residual connection
        # This means that we learn a new output layer norm, which replaces another layer norm learned in the bert layer
        if self.add_layer_norm_after:
            self.adapter_norm_after = MetaLayerNormLayer(self.input_size)

        # if we want to initialize with the bert strategy then this function is called for all the linear layers
        # if init_bert_weights:
        #     self.adapter_down.apply(self.init_bert_weights)
        #     self.adapter_up.apply(self.init_bert_weights)

    def forward(self, x, params=None):  # , residual_input=None):

        adapter_down_params = None
        adapter_up_params = None

        if params is not None:
            params = extract_top_level_dict(current_dict=params)
            adapter_down_params = extract_top_level_dict(params["adapter_down"])["0"]
            adapter_up_params = params["adapter_up"]

        # hard-coded
        # Linearity
        x = self.adapter_down[0](
            x, params=adapter_down_params if adapter_down_params is not None else None
        )
        # Activation fn
        x = self.adapter_down[1](x)

        output = self.adapter_up(x, params=adapter_up_params)

        # TODO: check config on layer norm settings

        # # apply residual connection before layer norm if configured in this way
        # if self.residual_before_ln:
        #     output = output + residual_input
        #
        # # apply layer norm if available
        # if self.add_layer_norm_after:
        #     output = self.adapter_norm_after(output)
        #
        # # if residual should be applied after layer norm, apply it here
        # if not self.residual_before_ln:
        #     output = output + residual_input

        return output


class MetaAdapterBertOutput(nn.Module):  # BertSelfOutputAdaptersMixin,
    def __init__(self, config, is_distil):
        super().__init__()
        self.config = config

        self.dense = MetaLinearLayer(config.intermediate_size, config.hidden_size)
        self.layer_norm_eps = 1e-12 if is_distil else config.layer_norm_eps
        self.LayerNorm = MetaLayerNormLayer(config.hidden_size, eps=self.layer_norm_eps)
        self.dropout_prob = config.hidden_dropout_prob
        self.dropout = nn.Dropout()

        self.adapter_fusion_layer = nn.ModuleDict(
            dict()
        )  # for compatibility with adapters library

        self.layer_text_task_adapters = nn.ModuleDict(
            {
                META_ADAPTER_NAME: MetaAdapterLayer(
                    input_size=config.hidden_size,
                    down_sample=config.adapter_down_sample_size,
                    non_linearity=config.adapters.config_map[
                        "text_task"
                    ].non_linearity.lower(),
                )
            }
        )
        self.layer_text_lang_adapters = nn.ModuleDict(
            dict()
        )  # for compatibility with adapters library

    def forward(self, hidden_states, input_tensor, num_step, params=None):
        dense_params = None
        layer_norm_params = None
        adapter_params = None
        if params is not None:
            params = extract_top_level_dict(current_dict=params)
            dense_params = params["dense"]
            layer_norm_params = params.get("LayerNorm", None)
            adapter_params = extract_top_level_dict(params["layer_text_task_adapters"])[
                META_ADAPTER_NAME
            ]

        hidden_states = self.dense(hidden_states, params=dense_params)

        if self.training:
            hidden_states = F.dropout(hidden_states, p=self.dropout_prob)
        # hidden_states = self.dropout(hidden_states)

        adapter = self.layer_text_task_adapters[META_ADAPTER_NAME]

        hidden_states = self.LayerNorm(
            hidden_states, num_step=num_step, params=layer_norm_params
        )
        hidden_states = adapter(hidden_states, params=adapter_params)

        return hidden_states + input_tensor


class MetaBertLayer(nn.Module):
    def __init__(self, config, is_distil, use_adapter=False):
        super().__init__()
        self.attention = MetaBertAttention(config, is_distil, use_adapter=use_adapter)

        self.intermediate = MetaBertIntermediate(config)
        if use_adapter:
            self.output = MetaAdapterBertOutput(config, is_distil)
        else:
            self.output = MetaBertOutput(config, is_distil)

    def forward(
        self,
        hidden_states,
        num_step,
        attention_mask=None,
        head_mask=None,
        # encoder_hidden_states=None,
        # encoder_attention_mask=None,
        params=None,
    ):
        attention_params = None
        intermediate_params = None
        output_params = None

        if params is not None:
            params = extract_top_level_dict(current_dict=params)
            attention_params = params["attention"]
            intermediate_params = params["intermediate"]
            output_params = params["output"]

        self_attention_outputs = self.attention(
            hidden_states=hidden_states,
            num_step=num_step,
            attention_mask=attention_mask,
            head_mask=head_mask,
            params=attention_params,
        )

        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[
            1:
        ]  # add self attentions if we output attention weights

        intermediate_output = self.intermediate(
            attention_output, params=intermediate_params
        )
        layer_output = self.output(
            hidden_states=intermediate_output,
            input_tensor=attention_output,
            num_step=num_step,
            params=output_params,
        )
        outputs = (layer_output,) + outputs
        return outputs


class MetaBertEncoder(nn.Module):
    def __init__(self, config, is_distil, use_adapter=False):
        super().__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList(
            [
                MetaBertLayer(config, is_distil, use_adapter=use_adapter)
                for _ in range(config.num_hidden_layers)
            ]
        )

    def forward(
        self,
        hidden_states,
        num_step,
        params=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        layer_params = {str(i): None for i in range(len(self.layer))}

        if params is not None:
            params = extract_top_level_dict(current_dict=params)
            layer_params = params["layer"]
            layer_params = extract_top_level_dict(layer_params)

        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states=hidden_states,
                num_step=num_step,
                attention_mask=attention_mask,
                head_mask=head_mask[i],
                # encoder_hidden_states=encoder_hidden_states,
                # encoder_attention_mask=encoder_attention_mask,
                params=layer_params[str(i)],
            )
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)


class MetaBertClassHead(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.out_features = config.num_labels
        self.dense = MetaLinearLayer(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        self.out_proj = MetaLinearLayer(config.hidden_size, config.num_labels)

    def forward(self, hidden_states, params=None, return_pooled=False):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        dense_params = None
        out_params = None

        if params is not None:
            params = extract_top_level_dict(current_dict=params)
            dense_params = params["dense"]
            out_params = params["out_proj"]

        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor, params=dense_params)
        pooled_output = self.activation(pooled_output)
        if return_pooled:
            return pooled_output

        logits = self.out_proj(pooled_output, params=out_params)

        return logits


class MetaConv2dLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        use_bias,
        groups=1,
        dilation_rate=1,
    ):
        """
        A MetaConv2D layer. Applies the same functionality of a standard Conv2D layer with the added functionality of
        being able to receive a parameter dictionary at the forward pass which allows the convolution to use external
        weights instead of the internal ones stored in the conv layer. Useful for inner loop optimization in the meta
        learning setting.
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param kernel_size: Convolutional kernel size
        :param stride: Convolutional stride
        :param padding: Convolution padding
        :param use_bias: Boolean indicating whether to use a bias or not.
        """
        super(MetaConv2dLayer, self).__init__()
        num_filters = out_channels
        self.stride = int(stride)
        self.padding = int(padding)
        self.dilation_rate = int(dilation_rate)
        self.use_bias = use_bias
        self.groups = int(groups)
        self.weight = nn.Parameter(
            torch.empty(num_filters, in_channels, kernel_size, kernel_size)
        )
        nn.init.xavier_uniform_(self.weight)

        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(num_filters))

    def forward(self, x, params=None):
        """
        Applies a conv2D forward pass. If params are not None will use the passed params as the conv weights and biases
        :param x: Input image batch.
        :param params: If none, then conv layer will use the stored self.weights and self.bias, if they are not none
        then the conv layer will use the passed params as its parameters.
        :return: The output of a convolutional function.
        """
        if params is not None:
            params = extract_top_level_dict(current_dict=params)
            if self.use_bias:
                (weight, bias) = params["weight"], params["bias"]
            else:
                (weight) = params["weight"]
                bias = None
        else:
            # print("No inner loop params")
            if self.use_bias:
                weight, bias = self.weight, self.bias
            else:
                weight = self.weight
                bias = None

        out = F.conv2d(
            input=x,
            weight=weight,
            bias=bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation_rate,
            groups=self.groups,
        )
        return out


class MetaLinearLayer(nn.Module):
    def __init__(self, input_shape, output_shape, use_bias=True):
        """
        A MetaLinear layer. Applies the same functionality of a standard linearlayer with the added functionality of
        being able to receive a parameter dictionary at the forward pass which allows the convolution to use external
        weights instead of the internal ones stored in the linear layer. Useful for inner loop optimization in the meta
        learning setting.
        :param input_shape: The shape of the input data, in the form (b, f)
        :param num_filters: Number of output filters
        :param use_bias: Whether to use biases or not.
        """
        super(MetaLinearLayer, self).__init__()

        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.ones(output_shape, input_shape))
        nn.init.xavier_uniform_(self.weight)
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(output_shape))
            nn.init.xavier_uniform_(self.weight)

    def forward(self, x, params=None):
        """
        Forward propagates by applying a linear function (Wx + b). If params are none then internal params are used.
        Otherwise passed params will be used to execute the function.
        :param x: Input data batch, in the form (b, f)
        :param params: A dictionary containing 'weights' and 'bias'. If params are none then internal params are used.
        Otherwise the external are used.
        :return: The result of the linear function.
        """
        weight, bias, = (
            None,
            None,
        )

        if params is not None:
            params = extract_top_level_dict(current_dict=params)
            if self.use_bias:
                (weight, bias) = params["weight"], params["bias"]
            else:
                (weight) = params["weight"]
                bias = None

        if weight is None:
            if self.use_bias:
                weight, bias = self.weight, self.bias
            else:
                weight = self.weight
                bias = None

        out = F.linear(input=x, weight=weight, bias=bias)
        return out


class MetaBatchNormLayer(nn.Module):
    def __init__(
        self,
        num_features,
        device,
        args,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
        meta_batch_norm=True,
        no_learnable_params=False,
        use_per_step_bn_statistics=False,
    ):
        """
        A MetaBatchNorm layer. Applies the same functionality of a standard BatchNorm layer with the added functionality of
        being able to receive a parameter dictionary at the forward pass which allows the convolution to use external
        weights instead of the internal ones stored in the conv layer. Useful for inner loop optimization in the meta
        learning setting. Also has the additional functionality of being able to store per step running stats and per step beta and gamma.
        :param num_features:
        :param device:
        :param args:
        :param eps:
        :param momentum:
        :param affine:
        :param track_running_stats:
        :param meta_batch_norm:
        :param no_learnable_params:
        :param use_per_step_bn_statistics:
        """
        super(MetaBatchNormLayer, self).__init__()
        self.num_features = num_features
        self.eps = eps

        self.affine = affine
        self.track_running_stats = track_running_stats
        self.meta_batch_norm = meta_batch_norm
        self.num_features = num_features
        self.device = device
        self.use_per_step_bn_statistics = use_per_step_bn_statistics
        self.args = args
        self.learnable_gamma = self.args.learnable_bn_gamma
        self.learnable_beta = self.args.learnable_bn_beta

        if use_per_step_bn_statistics:
            self.running_mean = nn.Parameter(
                torch.zeros(args.number_of_training_steps_per_iter, num_features),
                requires_grad=False,
            )
            self.running_var = nn.Parameter(
                torch.ones(args.number_of_training_steps_per_iter, num_features),
                requires_grad=False,
            )
            self.bias = nn.Parameter(
                torch.zeros(args.number_of_training_steps_per_iter, num_features),
                requires_grad=self.learnable_beta,
            )
            self.weight = nn.Parameter(
                torch.ones(args.number_of_training_steps_per_iter, num_features),
                requires_grad=self.learnable_gamma,
            )
        else:
            self.running_mean = nn.Parameter(
                torch.zeros(num_features), requires_grad=False
            )
            self.running_var = nn.Parameter(
                torch.zeros(num_features), requires_grad=False
            )
            self.bias = nn.Parameter(
                torch.zeros(num_features), requires_grad=self.learnable_beta
            )
            self.weight = nn.Parameter(
                torch.ones(num_features), requires_grad=self.learnable_gamma
            )

        if self.args.enable_inner_loop_optimizable_bn_params:
            self.bias = nn.Parameter(
                torch.zeros(num_features), requires_grad=self.learnable_beta
            )
            self.weight = nn.Parameter(
                torch.ones(num_features), requires_grad=self.learnable_gamma
            )

        self.backup_running_mean = torch.zeros(self.running_mean.shape)
        self.backup_running_var = torch.ones(self.running_var.shape)

        self.momentum = momentum

    def forward(
        self,
        input,
        num_step,
        params=None,
        training=False,
        backup_running_statistics=False,
    ):
        """
        Forward propagates by applying a bach norm function. If params are none then internal params are used.
        Otherwise passed params will be used to execute the function.
        :param input: input data batch, size either can be any.
        :param num_step: The current inner loop step being taken. This is used when we are learning per step params and
         collecting per step batch statistics. It indexes the correct object to use for the current time-step
        :param params: A dictionary containing 'weight' and 'bias'.
        :param training: Whether this is currently the training or evaluation phase.
        :param backup_running_statistics: Whether to backup the running statistics. This is used
        at evaluation time, when after the pass is complete we want to throw away the collected validation stats.
        :return: The result of the batch norm operation.
        """
        if params is not None:
            params = extract_top_level_dict(current_dict=params)
            (weight, bias) = params["weight"], params["bias"]
            # print(num_step, params['weight'])
        else:
            # print(num_step, "no params")
            weight, bias = self.weight, self.bias

        if self.use_per_step_bn_statistics:
            running_mean = self.running_mean[num_step]
            running_var = self.running_var[num_step]
            if params is None:
                if not self.args.enable_inner_loop_optimizable_bn_params:
                    bias = self.bias[num_step]
                    weight = self.weight[num_step]
        else:
            running_mean = None
            running_var = None

        if backup_running_statistics and self.use_per_step_bn_statistics:
            self.backup_running_mean.data = copy(self.running_mean.data)
            self.backup_running_var.data = copy(self.running_var.data)

        momentum = self.momentum

        output = F.batch_norm(
            input,
            running_mean,
            running_var,
            weight,
            bias,
            training=True,
            momentum=momentum,
            eps=self.eps,
        )

        return output

    def restore_backup_stats(self):
        """
        Resets batch statistics to their backup values which are collected after each forward pass.
        """
        if self.use_per_step_bn_statistics:
            self.running_mean = nn.Parameter(
                self.backup_running_mean.to(device=self.device), requires_grad=False
            )
            self.running_var = nn.Parameter(
                self.backup_running_var.to(device=self.device), requires_grad=False
            )

    def extra_repr(self):
        return (
            "{num_features}, eps={eps}, momentum={momentum}, affine={affine}, "
            "track_running_stats={track_running_stats}".format(**self.__dict__)
        )


class MetaLayerNormLayer(nn.Module):
    def __init__(self, input_feature_shape, eps=1e-5, elementwise_affine=True):
        """
        A MetaLayerNorm layer. A layer that applies the same functionality as a layer norm layer with the added
        capability of being able to receive params at inference time to use instead of the internal ones. As well as
        being able to use its own internal weights.
        :param input_feature_shape: The input shape without the batch dimension, e.g. c, h, w
        :param eps: Epsilon to use for protection against overflows
        :param elementwise_affine: Whether to learn a multiplicative interaction parameter 'w' in addition to
        the biases.
        """
        super(MetaLayerNormLayer, self).__init__()
        if isinstance(input_feature_shape, numbers.Integral):
            input_feature_shape = (input_feature_shape,)
        self.normalized_shape = input_feature_shape  # torch.Size(input_feature_shape)
        # print(type(self.normalized_shape))
        # print(type(input_feature_shape))
        # print(input_feature_shape)
        # input('her')
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.Tensor(*input_feature_shape))
            self.bias = nn.Parameter(torch.Tensor(*input_feature_shape))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        self.reset_parameters()

        self.num_steps = None

    def reset_parameters(self):
        """
        Reset parameters to their initialization values.
        """
        if self.elementwise_affine:
            self.weight.data.fill_(1)
            self.bias.data.zero_()

    def init_per_step_weights(self, num_steps):
        """
        Creates a copy of the current weights per step
        :return:
        """

        self.weight = torch.nn.Parameter(self.weight.repeat(num_steps, 1))
        self.bias = torch.nn.Parameter(self.bias.repeat(num_steps, 1))

        self.num_steps = num_steps

    def forward(
        self,
        input,
        num_step,
        params=None,
        training=False,
        backup_running_statistics=False,
    ):
        """
        Forward propagates by applying a layer norm function. If params are none then internal params are used.
        Otherwise passed params will be used to execute the function.
        :param input: input data batch, size either can be any.
        :param num_step: The current inner loop step being taken. This is used when we are learning per step params and
         collecting per step batch statistics. It indexes the correct object to use for the current time-step
        :param params: A dictionary containing 'weight' and 'bias'.
        :param training: Whether this is currently the training or evaluation phase.
        :param backup_running_statistics: Whether to backup the running statistics. This is used
        at evaluation time, when after the pass is complete we want to throw away the collected validation stats.
        :return: The result of the batch norm operation.
        """

        weight = None
        bias = None
        if params is not None:
            params = extract_top_level_dict(current_dict=params)
            weight = params["weight"]
            bias = params["bias"]

        if weight is None:
            weight = self.weight
            bias = self.bias

        if self.num_steps:
            weight = weight[num_step]
            bias = bias[num_step]

        return F.layer_norm(input, self.normalized_shape, weight, bias, self.eps)

    def restore_backup_stats(self):
        pass

    def extra_repr(self):
        return (
            "{normalized_shape}, eps={eps}, "
            "elementwise_affine={elementwise_affine}".format(**self.__dict__)
        )


class MetaConvNormLayerReLU(nn.Module):
    def __init__(
        self,
        input_shape,
        num_filters,
        kernel_size,
        stride,
        padding,
        use_bias,
        args,
        normalization=True,
        meta_layer=True,
        no_bn_learnable_params=False,
        device=None,
    ):
        """
        Initializes a BatchNorm->Conv->ReLU layer which applies those operation in that order.
        :param args: A named tuple containing the system's hyperparameters.
        :param device: The device to run the layer on.
        :param normalization: The type of normalization to use 'batch_norm' or 'layer_norm'
        :param meta_layer: Whether this layer will require meta-layer capabilities such as meta-batch norm,
        meta-conv etc.
        :param input_shape: The image input shape in the form (b, c, h, w)
        :param num_filters: number of filters for convolutional layer
        :param kernel_size: the kernel size of the convolutional layer
        :param stride: the stride of the convolutional layer
        :param padding: the bias of the convolutional layer
        :param use_bias: whether the convolutional layer utilizes a bias
        """
        super(MetaConvNormLayerReLU, self).__init__()
        self.normalization = normalization
        self.use_per_step_bn_statistics = args.per_step_bn_statistics
        self.input_shape = input_shape
        self.args = args
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_bias = use_bias
        self.meta_layer = meta_layer
        self.no_bn_learnable_params = no_bn_learnable_params
        self.device = device
        self.layer_dict = nn.ModuleDict()
        self.build_block()

    def build_block(self):

        x = torch.zeros(self.input_shape)

        out = x

        self.conv = MetaConv2dLayer(
            in_channels=out.shape[1],
            out_channels=self.num_filters,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            use_bias=self.use_bias,
        )

        out = self.conv(out)

        if self.normalization:
            if self.args.norm_layer == "batch_norm":
                self.norm_layer = MetaBatchNormLayer(
                    out.shape[1],
                    track_running_stats=True,
                    meta_batch_norm=self.meta_layer,
                    no_learnable_params=self.no_bn_learnable_params,
                    device=self.device,
                    use_per_step_bn_statistics=self.use_per_step_bn_statistics,
                    args=self.args,
                )
            elif self.args.norm_layer == "layer_norm":
                self.norm_layer = MetaLayerNormLayer(input_feature_shape=out.shape[1:])

            out = self.norm_layer(out, num_step=0)

        out = F.leaky_relu(out)

        print(out.shape)

    def forward(
        self, x, num_step, params=None, training=False, backup_running_statistics=False
    ):
        """
        Forward propagates by applying the function. If params are none then internal params are used.
        Otherwise passed params will be used to execute the function.
        :param input: input data batch, size either can be any.
        :param num_step: The current inner loop step being taken. This is used when we are learning per step params and
         collecting per step batch statistics. It indexes the correct object to use for the current time-step
        :param params: A dictionary containing 'weight' and 'bias'.
        :param training: Whether this is currently the training or evaluation phase.
        :param backup_running_statistics: Whether to backup the running statistics. This is used
        at evaluation time, when after the pass is complete we want to throw away the collected validation stats.
        :return: The result of the batch norm operation.
        """
        batch_norm_params = None
        conv_params = None
        activation_function_pre_params = None

        if params is not None:
            params = extract_top_level_dict(current_dict=params)

            if self.normalization:
                if "norm_layer" in params:
                    batch_norm_params = params["norm_layer"]

                if "activation_function_pre" in params:
                    activation_function_pre_params = params["activation_function_pre"]

            conv_params = params["conv"]

        out = x

        out = self.conv(out, params=conv_params)

        if self.normalization:
            out = self.norm_layer.forward(
                out,
                num_step=num_step,
                params=batch_norm_params,
                training=training,
                backup_running_statistics=backup_running_statistics,
            )

        out = F.leaky_relu(out)

        return out

    def restore_backup_stats(self):
        """
        Restore stored statistics from the backup, replacing the current ones.
        """
        if self.normalization:
            self.norm_layer.restore_backup_stats()


class MetaNormLayerConvReLU(nn.Module):
    def __init__(
        self,
        input_shape,
        num_filters,
        kernel_size,
        stride,
        padding,
        use_bias,
        args,
        normalization=True,
        meta_layer=True,
        no_bn_learnable_params=False,
        device=None,
    ):
        """
        Initializes a BatchNorm->Conv->ReLU layer which applies those operation in that order.
        :param args: A named tuple containing the system's hyperparameters.
        :param device: The device to run the layer on.
        :param normalization: The type of normalization to use 'batch_norm' or 'layer_norm'
        :param meta_layer: Whether this layer will require meta-layer capabilities such as meta-batch norm,
        meta-conv etc.
        :param input_shape: The image input shape in the form (b, c, h, w)
        :param num_filters: number of filters for convolutional layer
        :param kernel_size: the kernel size of the convolutional layer
        :param stride: the stride of the convolutional layer
        :param padding: the bias of the convolutional layer
        :param use_bias: whether the convolutional layer utilizes a bias
        """
        super(MetaNormLayerConvReLU, self).__init__()
        self.normalization = normalization
        self.use_per_step_bn_statistics = args.per_step_bn_statistics
        self.input_shape = input_shape
        self.args = args
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_bias = use_bias
        self.meta_layer = meta_layer
        self.no_bn_learnable_params = no_bn_learnable_params
        self.device = device
        self.layer_dict = nn.ModuleDict()
        self.build_block()

    def build_block(self):

        x = torch.zeros(self.input_shape)

        out = x
        if self.normalization:
            if self.args.norm_layer == "batch_norm":
                self.norm_layer = MetaBatchNormLayer(
                    self.input_shape[1],
                    track_running_stats=True,
                    meta_batch_norm=self.meta_layer,
                    no_learnable_params=self.no_bn_learnable_params,
                    device=self.device,
                    use_per_step_bn_statistics=self.use_per_step_bn_statistics,
                    args=self.args,
                )
            elif self.args.norm_layer == "layer_norm":
                self.norm_layer = MetaLayerNormLayer(input_feature_shape=out.shape[1:])

            out = self.norm_layer.forward(out, num_step=0)
        self.conv = MetaConv2dLayer(
            in_channels=out.shape[1],
            out_channels=self.num_filters,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            use_bias=self.use_bias,
        )

        self.layer_dict["activation_function_pre"] = nn.LeakyReLU()

        out = self.layer_dict["activation_function_pre"].forward(self.conv.forward(out))
        print(out.shape)

    def forward(
        self, x, num_step, params=None, training=False, backup_running_statistics=False
    ):
        """
        Forward propagates by applying the function. If params are none then internal params are used.
        Otherwise passed params will be used to execute the function.
        :param input: input data batch, size either can be any.
        :param num_step: The current inner loop step being taken. This is used when we are learning per step params and
         collecting per step batch statistics. It indexes the correct object to use for the current time-step
        :param params: A dictionary containing 'weight' and 'bias'.
        :param training: Whether this is currently the training or evaluation phase.
        :param backup_running_statistics: Whether to backup the running statistics. This is used
        at evaluation time, when after the pass is complete we want to throw away the collected validation stats.
        :return: The result of the batch norm operation.
        """
        batch_norm_params = None

        if params is not None:
            params = extract_top_level_dict(current_dict=params)

            if self.normalization:
                if "norm_layer" in params:
                    batch_norm_params = params["norm_layer"]

            conv_params = params["conv"]
        else:
            conv_params = None
            # print('no inner loop params', self)

        out = x

        if self.normalization:
            out = self.norm_layer.forward(
                out,
                num_step=num_step,
                params=batch_norm_params,
                training=training,
                backup_running_statistics=backup_running_statistics,
            )

        out = self.conv.forward(out, params=conv_params)
        out = self.layer_dict["activation_function_pre"].forward(out)

        return out

    def restore_backup_stats(self):
        """
        Restore stored statistics from the backup, replacing the current ones.
        """
        if self.normalization:
            self.norm_layer.restore_backup_stats()


class VGGReLUNormNetwork(nn.Module):
    def __init__(
        self, im_shape, num_output_classes, args, device, meta_classifier=True
    ):
        """
        Builds a multilayer convolutional network. It also provides functionality for passing external parameters to be
        used at inference time. Enables inner loop optimization readily.
        :param im_shape: The input image batch shape.
        :param num_output_classes: The number of output classes of the network.
        :param args: A named tuple containing the system's hyperparameters.
        :param device: The device to run this on.
        :param meta_classifier: A flag indicating whether the system's meta-learning (inner-loop) functionalities should
        be enabled.
        """
        super(VGGReLUNormNetwork, self).__init__()
        b, c, self.h, self.w = im_shape
        self.device = device
        self.total_layers = 0
        self.args = args
        self.upscale_shapes = []
        self.cnn_filters = args.cnn_num_filters
        self.input_shape = list(im_shape)
        self.num_stages = args.num_stages
        self.num_output_classes = num_output_classes

        if args.max_pooling:
            print("Using max pooling")
            self.conv_stride = 1
        else:
            print("Using strided convolutions")
            self.conv_stride = 2
        self.meta_classifier = meta_classifier

        self.build_network()
        print("meta network params")
        for name, param in self.named_parameters():
            print(name, param.shape)

    def build_network(self):
        """
        Builds the network before inference is required by creating some dummy inputs with the same input as the
        self.im_shape tuple. Then passes that through the network and dynamically computes input shapes and
        sets output shapes for each layer.
        """
        x = torch.zeros(self.input_shape)
        out = x
        self.layer_dict = nn.ModuleDict()
        self.upscale_shapes.append(x.shape)

        for i in range(self.num_stages):
            self.layer_dict["conv{}".format(i)] = MetaConvNormLayerReLU(
                input_shape=out.shape,
                num_filters=self.cnn_filters,
                kernel_size=3,
                stride=self.conv_stride,
                padding=self.args.conv_padding,
                use_bias=True,
                args=self.args,
                normalization=True,
                meta_layer=self.meta_classifier,
                no_bn_learnable_params=False,
                device=self.device,
            )
            out = self.layer_dict["conv{}".format(i)](out, training=True, num_step=0)

            if self.args.max_pooling:
                out = F.max_pool2d(input=out, kernel_size=(2, 2), stride=2, padding=0)

        if not self.args.max_pooling:
            out = F.avg_pool2d(out, out.shape[2])

        self.encoder_features_shape = list(out.shape)
        out = out.view(out.shape[0], -1)

        self.layer_dict["linear"] = MetaLinearLayer(
            input_shape=(out.shape[0], np.prod(out.shape[1:])),
            num_filters=self.num_output_classes,
            use_bias=True,
        )

        out = self.layer_dict["linear"](out)
        print("VGGNetwork build", out.shape)

    def forward(
        self, x, num_step, params=None, training=False, backup_running_statistics=False
    ):
        """
        Forward propages through the network. If any params are passed then they are used instead of stored params.
        :param x: Input image batch.
        :param num_step: The current inner loop step number
        :param params: If params are None then internal parameters are used. If params are a dictionary with keys the
         same as the layer names then they will be used instead.
        :param training: Whether this is training (True) or eval time.
        :param backup_running_statistics: Whether to backup the running statistics in their backup store. Which is
        then used to reset the stats back to a previous state (usually after an eval loop, when we want to throw away stored statistics)
        :return: Logits of shape b, num_output_classes.
        """
        param_dict = dict()

        if params is not None:
            param_dict = extract_top_level_dict(current_dict=params)

        # print('top network', param_dict.keys())
        for name, param in self.layer_dict.named_parameters():
            path_bits = name.split(".")
            layer_name = path_bits[0]
            if layer_name not in param_dict:
                param_dict[layer_name] = None

        out = x

        for i in range(self.num_stages):
            out = self.layer_dict["conv{}".format(i)](
                out,
                params=param_dict["conv{}".format(i)],
                training=training,
                backup_running_statistics=backup_running_statistics,
                num_step=num_step,
            )
            if self.args.max_pooling:
                out = F.max_pool2d(input=out, kernel_size=(2, 2), stride=2, padding=0)

        if not self.args.max_pooling:
            out = F.avg_pool2d(out, out.shape[2])

        out = out.view(out.size(0), -1)
        out = self.layer_dict["linear"](out, param_dict["linear"])

        return out

    def zero_grad(self, params=None):
        if params is None:
            for param in self.parameters():
                if param.requires_grad == True:
                    if param.grad is not None:
                        if torch.sum(param.grad) > 0:
                            print(param.grad)
                            param.grad.zero_()
        else:
            for name, param in params.items():
                if param.requires_grad == True:
                    if param.grad is not None:
                        if torch.sum(param.grad) > 0:
                            print(param.grad)
                            param.grad.zero_()
                            params[name].grad = None

    def restore_backup_stats(self):
        """
        Reset stored batch statistics from the stored backup.
        """
        for i in range(self.num_stages):
            self.layer_dict["conv{}".format(i)].restore_backup_stats()
