from transformers import Qwen2_5_VLForConditionalGeneration

from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLCausalLMOutputWithPast

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.nn import CrossEntropyLoss, LayerNorm

IGNORE_INDEX = -100

# reward_func_v1 = lambda dist, ans: torch.exp(-dist**2 / (0.1 * (ans + 0.1)))
# reward_func_v1 = lambda dist, ans: torch.exp(-dist**2 / (0.1 * (ans + 0.1)))

def reward_func_v0(dist, sigma = 1):
    dist = torch.exp(-torch.square(dist) / (2 * sigma ** 2))
    return dist


class Qwen2_5_VLForConditionalGenerationSpecialToken(Qwen2_5_VLForConditionalGeneration):

    def grounding_token_soft_ce_loss(self, logits, labels):
        grounding_token_ids = self.config.grounding_token_ids
        grounding_token_start_id = min(grounding_token_ids)
        grounding_token_end_id = max(grounding_token_ids)

        logits = logits.float()
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # # Flatten the tokens
        loss_fct = CrossEntropyLoss(reduction="none")
        # print(shift_labels.shape, shift_logits.shape)
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)

        valid_indices = (shift_labels != IGNORE_INDEX)
        shift_logits = shift_logits[valid_indices]
        shift_labels = shift_labels[valid_indices]

        grounding_mask = torch.logical_and(shift_labels <= grounding_token_end_id, shift_labels >= grounding_token_start_id)

        shift_labels_grounding = shift_labels[grounding_mask]
        shift_logits_grounding = shift_logits[grounding_mask]

        shift_labels_normal = shift_labels[~grounding_mask]
        shift_logits_normal = shift_logits[~grounding_mask]
        loss_normal = loss_fct(shift_logits_normal, shift_labels_normal)

        gt_labels_groundings = torch.zeros((len(shift_labels_grounding), self.config.vocab_size), dtype=torch.float).to(shift_logits.device)

        for idx, label in enumerate(shift_labels_grounding):
            # if label not in grounding_token_ids:
            #     gt_labels[idx][label] = 1
            # else:
            dist = label - torch.Tensor(grounding_token_ids).to(label)
            # dist = torch.exp(-torch.square(dist) / (2 * std_dev ** 2))
            if self.config.reward_func == 'v0':
                dist = reward_func_v0(dist)
            else:
                raise NotImplementedError

            dist = dist / dist.sum()
            # print(label - grounding_token_start_id, label, dist)
            gt_labels_groundings[idx][grounding_token_ids] = dist * self.config.grounding_token_ce_lambda

        # loss = loss_fct(shift_logits, shift_labels)
        loss_grounding = loss_fct(shift_logits_grounding, gt_labels_groundings)
        # print(loss_grounding.shape, loss_normal.shape)
        loss = torch.cat([loss_grounding, loss_normal], dim=-1).mean()
        # import pdb; pdb.set_trace()
        return loss

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:

        outputs = super().forward(
            input_ids = input_ids,
            attention_mask = attention_mask,
            position_ids = position_ids,
            past_key_values = past_key_values,
            inputs_embeds = inputs_embeds,
            labels = labels,
            use_cache = use_cache,
            output_attentions = output_attentions,
            output_hidden_states = output_hidden_states,
            return_dict = return_dict,
            pixel_values = pixel_values,
            pixel_values_videos = pixel_values_videos,
            image_grid_thw = image_grid_thw,
            video_grid_thw = video_grid_thw,
            rope_deltas = rope_deltas,
            cache_position = cache_position,
            second_per_grid_ts = second_per_grid_ts
        )

        if not self.training or labels is None:
            return outputs


        loss_new = None
        if not return_dict:
            loss_old = outputs[0]
            logits = outputs[1]
        else:
            loss_old = outputs.loss
            logits = outputs.logits

        loss_new = self.grounding_token_soft_ce_loss(logits, labels) + loss_old * 0.0

        if not return_dict:
            outputs = (loss_new,) + outputs[1:]
            return outputs

        outputs.loss = loss_new
        return outputs


if __name__ == "__main__":
    model = Qwen2_5_VLForConditionalGenerationSpecialToken.from_pretrained(
        "Qwen/Qwen2。5-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
    )
    from qwen_vl_utils import process_vision_info