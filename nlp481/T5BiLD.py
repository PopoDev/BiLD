"""
This is the T5 BiLD implementation as provided in
https://github.com/kssteven418/BigLittleDecoder
"""

from typing import Optional, Tuple
from torch import nn
from transformers.generation_utils import GenerationMixin

import copy
import torch

####################################################
# This dict contains ids and associated url
# for the pretrained weights provided with the models
####################################################
T5_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "t5-small",
    "t5-base",
    "t5-large",
    "t5-3b",
    "t5-11b",
    # See all T5 models at https://huggingface.co/models?filter=t5
]

class T5BiLDModel(nn.Module, GenerationMixin):
    def __init__(
        self, 
        large, 
        small, 
        num_small_iters=10, 
        fallback_threshold=0.6,
        rollback_threshold=5.0,
    ):
        super().__init__()
        self.large = large # Large T5 model
        self.small = small # Small T5 model

        self.num_large_iters = 1
        # defines the maximum possible number of small model iterations
        self.num_small_iters = num_small_iters

        self.init_iters(init_with='large')

        self.encoder = self.get_encoder()
        self.decoder = self.get_decoder()
        self.main_input_name = self.large.main_input_name

        self.fallback_threshold = fallback_threshold or 0.6
        self.rollback_threshold = rollback_threshold or 5.0

        self.crossentropy_loss = nn.CrossEntropyLoss(reduce=False)

    def is_large(self):
        return self.model_type == 'large'

    def get_encoder(self):
        return self.large.encoder if self.is_large() else self.small.encoder

    def get_decoder(self):
        return self.large.decoder if self.is_large() else self.small.decoder

    def init_iters(self, model_kwargs=None, init_with='large'):
        """
        initiate the dual model.
        Args:
            model_kwargs (dict): model_kwargs
                - deep copied as small and large models' model_kwargs separately
            init_with (str): whether to init with large or small model
        """
        assert init_with in ['large', 'small']

        self.model_type = init_with
        self.iter_count = self.num_large_iters

        self.large_kwargs = copy.deepcopy(model_kwargs)
        self.small_kwargs = copy.deepcopy(model_kwargs)

        if model_kwargs is not None:
            # replace small model encoder output
            self.small_kwargs.pop("encoder_outputs_small")
            self.small_kwargs["encoder_outputs"] = self.large_kwargs.pop("encoder_outputs_small")

        if init_with == 'large':
            self.model_kwargs = self.large_kwargs
        else: # small
            self.model_kwargs = self.small_kwargs

    def schedule_iters(self, fall_back_to_large=False, fall_back_to_small=False):
        """
        schedule large and small models.
        Args:
            fall_back_to_large (bool): force fall back from small to large model
                otherwise, hand over the control when the iteration counts become 0
        """
        self.iter_count -= 1

        to_small = self.is_large() and (self.iter_count == 0 or fall_back_to_small)
        to_large = not self.is_large() and (self.iter_count == 0 or fall_back_to_large)

        if to_small:
            self.model_type = 'small'
            self.iter_count = self.num_small_iters
            self.model_kwargs = self.small_kwargs

        if to_large:
            self.model_type = 'large'
            self.iter_count = self.num_large_iters
            self.model_kwargs = self.large_kwargs

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """Runs the forward function of either the large model or the small model"""
        args = [
            input_ids,
            attention_mask,
            decoder_input_ids,
            decoder_attention_mask,
            head_mask,
            decoder_head_mask,
            cross_attn_head_mask,
            encoder_outputs,
            past_key_values,
            inputs_embeds,
            decoder_inputs_embeds,
            labels,
            use_cache,
            output_attentions,
            output_hidden_states,
            return_dict,
        ]
        if self.is_large():
            return self.large(*args)
        else:
            return self.small(*args)

    @property
    def config(self):
        return self.large.config

    def resize_token_embeddings(self, n):
        self.large.resize_token_embeddings(n)
        self.small.resize_token_embeddings(n)

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            previous_generated_len = past[0][0].shape[2]
            input_ids = input_ids[:, previous_generated_len:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

    def _reset_kwargs_past_to_new_length(self, new_len):
        """
        reset both small and large model kwargs past into the given length
        """
        for kwargs in [self.large_kwargs, self.small_kwargs]:
            new_kwargs = []
            for layer_past in kwargs['past']:
                new_layer_kwargs = []
                for i, past in enumerate(layer_past):
                    if i < 2:
                        new_layer_kwargs.append(past[:, :, :new_len - 1, :])
                    else:
                        new_layer_kwargs.append(past)
                new_kwargs.append(tuple(new_layer_kwargs))
            kwargs['past'] = tuple(new_kwargs)

    def _greedy_search_body(
        self,
        input_ids,
        model_kwargs,
        output_attentions,
        output_hidden_states,
        stopping_criteria,
        logits_processor,
        pad_token_id,
        eos_token_id,
        synced_gpus,
        unfinished_sequences,
    ):
        assert not synced_gpus

        self.init_iters(model_kwargs=model_kwargs, init_with='large')
        scores = None
        self.rollback_signal = None

        while True:
            # Iteration right after the rollback
            # need to remove previous k and v caches for the rolled back tokens
            if self.rollback_signal:
                new_len = input_ids.shape[-1]
                self._reset_kwargs_past_to_new_length(new_len)
                self.rollback_signal = None

            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **self.model_kwargs)

            # past_key_values: #layer list,
            # each element is dict {'self', 'encoder_decoder'}
            # each has 'prev_key' and 'previous_value'
            # for 'self' they grow in sequence length
            # for 'encoder_decoder' the sequence length is fixed

            # forward pass to get next token
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            next_token_logits = outputs.logits[:, -1, :]

            # pre-process distribution
            next_tokens_scores = logits_processor(input_ids, next_token_logits)
            score = torch.softmax(next_tokens_scores, dim=-1)

            # argmax policy for the next token
            next_tokens = torch.argmax(score, dim=-1)

            # Fallback condition
            fallback_cond = (
                self.model_type == 'small'
                and score.max() < self.fallback_threshold
            )

            if fallback_cond:
                # if fall back, we ignore the current run
                # the large model will produce the same token (i.e. redundant)
                self.schedule_iters(fall_back_to_large=True)
                continue

            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)


            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

            # If running with the large model, check whether we want to rollback the small model's predictions
            if not self.training and self.is_large():
                large_model_logits = outputs.logits[0, :, :]
                if large_model_logits.shape[0] != 1:
                    # Compare the small model's predictions so far vs. the large model's non-autoregressive predictions
                    small_model_prediction = model_inputs["decoder_input_ids"][0]
                    large_model_prediction = large_model_logits.argmax(-1)

                    small_model_prediction = small_model_prediction[1:] # SL-1
                    large_model_logits = large_model_logits[:-1, :] # SL-1 x Dim

                    loss = self.crossentropy_loss(large_model_logits, small_model_prediction)
                    loss_above_thres = loss > self.rollback_threshold

                    # if there exists any predictions that deviates above threshold vs. the large model's prediction
                    if loss_above_thres.any():
                        # get the earliest index among those above-threshold prediction
                        first_idx_loss_above_thres = loss_above_thres.to(torch.int).argmax()
                        past = model_inputs['past_key_values']
                        past_len = past[0][0].shape[2]
                        new_len = first_idx_loss_above_thres + past_len + 1
                        new_input_ids = input_ids[:, :new_len]

                        new_pred = nn.functional.softmax(
                            large_model_logits[first_idx_loss_above_thres:first_idx_loss_above_thres + 1, :],
                            dim=-1,
                        ).argmax(-1).unsqueeze(0)

                        # Minor optimization:
                        # Avoid rollback if the new prediction from the large model is same as the small model's old prediction
                        # You can remove this condition
                        if new_pred[0] != input_ids[0, new_len]:
                            input_ids = torch.cat([new_input_ids, new_pred], dim=-1)
                            self.rollback_signal = True
                            continue

            self.model_kwargs = self._update_model_kwargs_for_generation(
                outputs, self.model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )

            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id is not None:
                unfinished_sequences = unfinished_sequences.mul((next_tokens != eos_token_id).long())

            # stop when each sentence is finished, or if we exceed the maximum length
            if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, scores):
                break

            self.schedule_iters()

        return input_ids


    def _prepare_encoder_decoder_kwargs_for_generation(
        self, inputs_tensor: torch.Tensor, model_kwargs, model_input_name: Optional[str] = None
    ):
        # 2. prepare encoder args and encoder kwargs from model kwargs
        irrelevant_prefix = ["decoder_", "cross_attn", "use_cache"]
        encoder_kwargs = {
            argument: value
            for argument, value in model_kwargs.items()
            if not any(argument.startswith(p) for p in irrelevant_prefix)
        }

        # 3. make sure that encoder returns `ModelOutput`
        model_input_name = model_input_name if model_input_name is not None else self.main_input_name
        encoder_kwargs["return_dict"] = True
        encoder_kwargs[model_input_name] = inputs_tensor
        #s = time.time()
        model_kwargs["encoder_outputs"] = self.large.encoder(**encoder_kwargs)
        model_kwargs["encoder_outputs_small"] = self.small.encoder(**encoder_kwargs)

        return model_kwargs
