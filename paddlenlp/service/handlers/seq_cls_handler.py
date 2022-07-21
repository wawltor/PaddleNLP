# coding:utf-8
# Copyright (c) 2022  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np
from .base_handler import BaseModelHandler
from ...transformers import AutoTokenizer
from ...data import Tuple, Pad


class SequenceClassificationModelHandler(BaseModelHandler):

    def __init__(self):
        super().__init__()

    @classmethod
    def process(cls, predictor, tokenizer, data, parameters):
        max_seq_len = 128
        batch_size = 1
        if 'max_seq_len' not in parameters:
            max_seq_len = parameters['max_seq_len']
        if 'batch_size' not in parameters:
            batch_size = parameters['max_seq_len']
        text = None
        if 'text' in data:
            text = data['text']
        if text is None:
            return {}
        text_pair = None
        if 'text_pair' in data:
            text_pair = data['text_pair']
        print("The text:{}".format(text))
        examples = []

        if isinstance(text, str):
            text = [text]
        if text_pair is not None:
            if isinstance(text_pair, str):
                text_pair = [text_pair]
        if text_pair is None:
            for data in text:
                result = tokenizer(text=data, max_length=max_seq_len)
                examples.append((result['input_ids'], result['token_type_ids']))
        else:
            for data1, data2 in zip(text, text_pair):
                result = tokenizer(text=data1,
                                   text_pair=data2,
                                   max_length=max_seq_len)
                examples.append((result['input_ids'], result['token_type_ids']))

        # Seperates data into some batches.
        batches = [
            examples[i:i + batch_size]
            for i in range(0, len(examples), batch_size)
        ]

        batchify_fn = lambda samples, fn=Tuple(
            Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype='int64'),  # input
            Pad(axis=0, pad_val=tokenizer.pad_token_type_id, dtype='int64'
                ),  # segment
        ): fn(samples)

        results = []
        for batch in batches:
            input_ids, token_type_ids = batchify_fn(batch)
            if predictor._predictor_type == 'paddle_inference':
                predictor._input_handles[0].copy_from_cpu(input_ids)
                predictor._input_handles[1].copy_from_cpu(token_type_ids)
                predictor._predictor.run()
                output = [
                    output_handle.copy_to_cpu()
                    for output_handle in predictor._output_handles
                ]
                results.extend(output)
            else:
                result = self.predictor._predictor.run(
                    None, {
                        'input_ids': input_ids,
                        'token_type_ids': token_type_ids
                    })
                results.extend(result)

        # Resolve the logits result and get the predict label and confidence
        results = np.concatenate(results, axis=0)
        max_value = np.max(results, axis=1, keepdims=True)
        exp_data = np.exp(results - max_value)
        probs = exp_data / np.sum(exp_data, axis=1, keepdims=True)
        out_dict = {
            "label": results.argmax(axis=-1).tolist(),
            "confidence": probs.max(axis=-1).tolist()
        }
        return out_dict
