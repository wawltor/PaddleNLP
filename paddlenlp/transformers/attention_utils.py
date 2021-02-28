#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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
import copy
import collections

from paddle import ParamAttr
import paddle
from paddle.nn import Linear, Dropout, LayerNorm, LayerList, Layer
import paddle.nn as nn
import paddle.nn.functional as F


class Registry(object):
    def __init__(self):
        self.cls_dict = {}

    def register(self, name):
        def add_item(name, cls):
            self.cls_dict[name] = cls
            return cls

        return lambda cls: add_item(name, cls)


AttentionRegistry = Registry()


class Mask(object):
    def __init__(self,
                 query_length,
                 key_length,
                 num_heads,
                 block_size,
                 window_size,
                 num_global_blocks,
                 num_rand_blocks,
                 seed=None):
        for k, v in locals().items():
            if k != "self":
                setattr(self, k, v)
        self.mask = np.zeros_like(
            np.arange(query_length * key_length * num_heads).reshape((
                num_heads, query_length, key_length)))
        self.rand_mask = np.zeros_like(
            np.arange(query_length * key_length * num_heads).reshape((
                num_heads, query_length, key_length)))
        self.rand_mask_idx = [[] for i in range(num_heads)]
        self.num_query_blocks = self.query_length // self.block_size     \
                + int(self.query_length % self.block_size != 0)
        self.num_key_blocks = self.key_length // self.block_size         \
                + int(self.key_length % self.block_size != 0)
        self.num_window_blocks = self.window_size // 2
        if seed:
            np.random.seed(seed)
        # create global mask
        self._create_global_mask()
        # create window mask
        self._create_window_mask()
        # create random mask
        self._create_random_mask()

    def get_mask(self):
        return self.mask

    def get_rand_mask_idx(self):
        return self.rand_mask_idx

    def get_rand_mask(self):
        return self.rand_mask

    def get_float_mask(self):
        float_mask = np.array(self.mask, dtype='float32')
        float_mask[float_mask != 1] = -np.inf
        float_mask[float_mask == 1.] = 0
        return float_mask

    def _create_global_mask(self):
        global_block_length = self.num_global_blocks * self.block_size
        self.mask[:, 0:global_block_length, :] = 1
        self.mask[:, :, 0:global_block_length] = 1

    def _create_window_mask(self):
        for query_block_idx in range(self.num_query_blocks):
            left_key_block_idx, right_key_block_idx = self._get_window_block_idx(
                query_block_idx)
            left_idx = left_key_block_idx * self.block_size
            right_idx = (right_key_block_idx + 1) * self.block_size
            query_left_idx = query_block_idx * self.block_size
            query_right_idx = min((query_block_idx + 1) * self.block_size,
                                  self.query_length)
            self.mask[:, query_left_idx:query_right_idx, left_idx:right_idx] = 1

    def _create_random_mask(self):
        all_key_blocks_idx = np.arange(0, self.num_key_blocks, dtype=np.int32)
        for query_block_idx in range(self.num_query_blocks):
            left_key_block_idx, right_key_block_idx = self._get_window_block_idx(
                query_block_idx)
            illegal_blocks_idx = [
                i for i in range(left_key_block_idx, right_key_block_idx + 1)
            ]
            illegal_blocks_idx.extend(
                [i for i in range(self.num_global_blocks)])
            left_key_block_idx = query_block_idx - self.num_window_blocks
            right_key_block_idx = query_block_idx + self.num_window_blocks
            if self.num_global_blocks > left_key_block_idx:
                num_fill_blocks = self.num_global_blocks - left_key_block_idx
                illegal_blocks_idx.extend([
                    i
                    for i in range(self.num_key_blocks - num_fill_blocks,
                                   self.num_key_blocks)
                ])
            if right_key_block_idx >= self.num_key_blocks:
                num_fill_blocks = right_key_block_idx - self.num_key_blocks + 1
                illegal_blocks_idx.extend([
                    i
                    for i in range(self.num_global_blocks,
                                   self.num_global_blocks + num_fill_blocks)
                ])

            illegal_blocks_idx = set(illegal_blocks_idx)

            query_left_idx = query_block_idx * self.block_size
            query_right_idx = min((query_block_idx + 1) * self.block_size,
                                  self.query_length)
            for i in range(self.num_heads):
                legal_blocks_idx = []
                legal_idx = []
                perm_block = np.random.permutation(all_key_blocks_idx)
                for j in perm_block:
                    if j not in illegal_blocks_idx:
                        legal_blocks_idx.append(j)
                    if len(legal_blocks_idx) == self.num_rand_blocks:
                        break
                for j in legal_blocks_idx:
                    key_left_idx = j * self.block_size
                    key_right_idx = min((j + 1) * self.block_size,
                                        self.key_length)
                    legal_idx.extend(
                        [i for i in range(key_left_idx, key_right_idx)])
                    self.rand_mask[i, query_left_idx:query_right_idx,
                                   key_left_idx:key_right_idx] = 1
                self.rand_mask_idx[i].append(legal_blocks_idx)
        self.rand_mask_idx = np.stack(self.rand_mask_idx, axis=0)
        self.rand_mask_idx = self.rand_mask_idx[:, self.num_global_blocks:]
        # transform rand_mask_idx
        H = self.rand_mask_idx.shape[0]
        L = self.rand_mask_idx.shape[1]
        R = self.rand_mask_idx.shape[2]
        rand_mask_idx = []
        for i in range(H):
            for j in range(L):
                for k in range(R):
                    rand_mask_idx.append([i, self.rand_mask_idx[i][j][k]])
        self.rand_mask_idx = rand_mask_idx
        self.mask = np.maximum(self.rand_mask, self.mask)

    def _get_window_block_idx(self, query_block_idx):
        left_key_block_idx = max(0, query_block_idx - self.num_window_blocks)
        right_key_block_idx = min(query_block_idx + self.num_window_blocks,
                                  self.num_key_blocks - 1)
        return left_key_block_idx, right_key_block_idx


def create_bigbird_simulated_attention_mask_list(
        num_layers, query_length, key_length, num_heads, block_size,
        window_size, num_global_blocks, num_rand_blocks, seed):
    attn_mask_list = []
    rand_mask_idx_list = []
    for i in range(num_layers):
        mask = Mask(query_length, key_length, num_heads, block_size,
                    window_size, num_global_blocks, num_rand_blocks, seed)
        attn_mask = paddle.to_tensor(mask.get_float_mask())
        rand_mask_idx = paddle.to_tensor(mask.get_rand_mask_idx())
        attn_mask_list.append(attn_mask)
        rand_mask_idx_list.append(rand_mask_idx)
    return attn_mask_list, rand_mask_idx_list


def create_bigbird_rand_mask_idx(num_layers, query_length, key_length,
                                 num_heads, block_size, window_size,
                                 num_global_blocks, num_rand_blocks, seed):
    #TODO(zsj): need to simplify
    num_key_blocks = key_length // block_size
    num_query_blocks = query_length // block_size
    num_window_blocks = window_size // 2
    all_key_blocks_idx = np.arange(0, num_key_blocks, dtype=np.int32)
    rand_mask_idx = [[] for i in range(num_heads)]
    for query_block_idx in range(num_query_blocks):
        left_key_block_idx = max(0, query_block_idx - num_window_blocks)
        right_key_block_idx = min(query_block_idx + num_window_blocks,
                                  num_key_blocks - 1)

        illegal_blocks_idx = [
            i for i in range(left_key_block_idx, right_key_block_idx + 1)
        ]
        illegal_blocks_idx.extend([i for i in range(num_global_blocks)])
        left_key_block_idx = query_block_idx - num_window_blocks
        right_key_block_idx = query_block_idx + num_window_blocks

        if num_global_blocks > left_key_block_idx:
            num_fill_blocks = num_global_blocks - left_key_block_idx
            illegal_blocks_idx.extend([
                i
                for i in range(num_key_blocks - num_fill_blocks, num_key_blocks)
            ])
        if right_key_block_idx >= num_key_blocks:
            num_fill_blocks = right_key_block_idx - num_key_blocks + 1
            illegal_blocks_idx.extend([
                i
                for i in range(num_global_blocks, num_global_blocks +
                               num_fill_blocks)
            ])

        illegal_blocks_idx = set(illegal_blocks_idx)

        for i in range(num_heads):
            legal_blocks_idx = []
            perm_block = np.random.permutation(all_key_blocks_idx)
            for j in perm_block:
                if j not in illegal_blocks_idx:
                    legal_blocks_idx.append(j)
                if len(legal_blocks_idx) == num_rand_blocks:
                    break
            rand_mask_idx[i].append(legal_blocks_idx)
    rand_mask_idx = np.stack(rand_mask_idx, axis=0)
    rand_mask_idx = rand_mask_idx[:,
                                  num_global_blocks:] - num_global_blocks // 2
    # transform rand_mask_idx
    H = rand_mask_idx.shape[0]
    L = rand_mask_idx.shape[1]
    R = rand_mask_idx.shape[2]
    rand_mask_idx = rand_mask_idx.reshape([-1, 1])
    head_idx = np.arange(H).reshape([-1, 1])
    head_idx = np.pad(head_idx, ([0, 0], [0, L * R - 1]),
                      mode="edge").reshape([-1, 1])
    rand_mask_idx_list = np.concatenate([head_idx, rand_mask_idx], axis=1)
    return rand_mask_idx_list


def create_bigbird_rand_mask_idx_list(num_layers, query_length, key_length,
                                      num_heads, block_size, window_size,
                                      num_global_blocks, num_rand_blocks, seed):
    rand_mask_idx_list = [
        create_bigbird_rand_mask_idx(num_layers, query_length, key_length,
                                     num_heads, block_size, window_size,
                                     num_global_blocks, num_rand_blocks, seed)
        for i in range(num_layers)
    ]
    rand_mask_idx_list = np.stack(rand_mask_idx_list)
    rand_mask_idx_list = paddle.to_tensor(rand_mask_idx_list)
    return rand_mask_idx_list


def _convert_param_attr_to_list(param_attr, n):
    if isinstance(param_attr, (list, tuple)):
        assert len(param_attr) == n, (
            "length of param_attr should be %d when it is a list/tuple" % n)
        param_attrs = []
        for attr in param_attr:
            if isinstance(attr, bool):
                if attr:
                    param_attrs.append(ParamAttr._to_attr(None))
                else:
                    param_attrs.append(False)
            else:
                param_attrs.append(ParamAttr._to_attr(attr))
    elif isinstance(param_attr, bool):
        param_attrs = []
        if param_attr:
            param_attrs = [ParamAttr._to_attr(None) for i in range(n)]
        else:
            param_attrs = [False] * n
    else:
        param_attrs = []
        attr = ParamAttr._to_attr(param_attr)
        for i in range(n):
            attr_i = copy.deepcopy(attr)
            if attr.name:
                attr_i.name = attr_i.name + "_" + str(i)
            param_attrs.append(attr_i)
    return param_attrs


class Linear3D(Layer):
    def __init__(self,
                 hidden_size,
                 num_attention_heads,
                 size_per_head,
                 weight_attr=None,
                 bias_attr=None):
        super(Linear3D, self).__init__()
        self._dtype = self._helper.get_default_dtype()
        self._weight_attr = weight_attr
        self._bias_attr = bias_attr
        self.weight = self.create_parameter(
            shape=[hidden_size, hidden_size],
            attr=self._weight_attr,
            dtype=self._dtype,
            is_bias=False)
        self.bias = self.create_parameter(
            shape=[hidden_size],
            attr=self._bias_attr,
            dtype=self._dtype,
            is_bias=True)
        self.size_per_head = size_per_head
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size

    def forward(self, input):
        # abc,cde->adbe
        reshape_input = paddle.unsqueeze(input, 1)
        reshape_w = paddle.reshape(
            self.weight,
            [self.hidden_size, self.num_attention_heads, self.size_per_head])
        reshape_w = paddle.transpose(reshape_w, [1, 0, 2])
        reshape_w = paddle.unsqueeze(reshape_w, 0)
        result = paddle.matmul(reshape_input, reshape_w)
        reshape_b = paddle.reshape(
            self.bias, [1, self.num_attention_heads, 1, self.size_per_head])
        result += reshape_b
        return result


class LinearProj3D(Layer):
    def __init__(self,
                 hidden_size,
                 num_attention_heads,
                 size_per_head,
                 weight_attr=None,
                 bias_attr=None):
        super(LinearProj3D, self).__init__()
        self._dtype = self._helper.get_default_dtype()
        self._weight_attr = weight_attr
        self._bias_attr = bias_attr
        self.weight = self.create_parameter(
            shape=[hidden_size, hidden_size],
            attr=self._weight_attr,
            dtype=self._dtype,
            is_bias=False)
        self.bias = self.create_parameter(
            shape=[hidden_size],
            attr=self._bias_attr,
            dtype=self._dtype,
            is_bias=True)
        self.size_per_head = size_per_head
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size

    def forward(self, input):
        # BFNH,NHD->BFD
        result = paddle.matmul(input, self.weight)


class Attention(Layer):
    def __init__(self,
                 num_heads=1,
                 block_size=1,
                 window_size=3,
                 num_global_blocks=1,
                 num_rand_blocks=1,
                 seed=None):
        super().__init__()

    def forward(self,
                query_matrix,
                key_matrix,
                value_matrix,
                d_head,
                attn_mask=None,
                rand_mask_idx=None,
                query_mask=None,
                key_mask=None,
                dropout=None):
        raise NotImplementedError


@AttentionRegistry.register("default_attention")
class DefaultAttention(Attention):
    def forward(self,
                query_matrix,
                key_matrix,
                value_matrix,
                d_head,
                attn_mask=None,
                rand_mask_idx=None,
                query_mask=None,
                key_mask=None,
                dropout=None):
        # scale dot product attention
        product = paddle.matmul(x=query_matrix, y=key_matrix, transpose_y=True)
        product = product * (d_head**-0.5)
        if attn_mask is not None:
            product = product + attn_mask
        weights = F.softmax(product)
        if dropout:
            weights = F.dropout(
                weights,
                dropout,
                training=self.training,
                mode="upscale_in_train")

        out = paddle.matmul(weights, value_matrix)
        return out


@AttentionRegistry.register("bigbird")
class BigBirdSparseAttention(Attention):
    def __init__(self,
                 num_heads=1,
                 block_size=1,
                 window_size=3,
                 num_global_blocks=1,
                 num_rand_blocks=1,
                 seed=None):
        super(BigBirdSparseAttention,
              self).__init__(num_heads, block_size, window_size,
                             num_global_blocks, num_rand_blocks, seed)
        for k, v in locals().items():
            if k != "self":
                setattr(self, k, v)
        self.num_global_blocks_back = num_global_blocks // 2
        self.num_global_blocks_front = num_global_blocks // 2   \
                if num_global_blocks % 2 == 0                  \
                else num_global_blocks // 2 + 1

    def _get_band_mask(self, blocked_query_mask, blocked_key_mask, batch_size,
                       sequence_length):
        '''
        Return second mask: [B, 1, L-G, bs, G+W]
        '''
        GB = self.num_global_blocks_back
        GF = self.num_global_blocks_front
        G = self.num_global_blocks
        R = self.num_rand_blocks
        W = self.window_size
        bs = self.block_size
        T = sequence_length
        L = T // bs  # blocked length
        B = batch_size
        H = self.num_heads
        # G+W+R
        # query_mask: [B, L, bs]
        # key_mask: [B, L, bs]
        # [B, L-G, bs, 1] * [B, L-G, 1, G*bs] -> [B, L-G, bs, G*bs]
        temp_query_mask = paddle.reshape(blocked_query_mask[:, GF:-GB],
                                         [B, L - G, bs, 1])
        temp_key_mask_front = paddle.reshape(blocked_key_mask[:, :GF],
                                             [B, 1, 1, GF * bs])
        temp_key_mask_front = paddle.expand(temp_key_mask_front,
                                            [B, L - G, 1, GF * bs])
        global_block_mask_front = paddle.matmul(temp_query_mask,
                                                temp_key_mask_front)

        temp_key_mask_back = paddle.reshape(blocked_key_mask[:, -GB:],
                                            [B, 1, 1, GB * bs])
        temp_key_mask_back = paddle.expand(temp_key_mask_back,
                                           [B, L - G, 1, GB * bs])
        global_block_mask_back = paddle.matmul(temp_query_mask,
                                               temp_key_mask_back)

        # create window block mask
        key_mask_list = []
        for query_block_id in range(GF, GF + W // 2):
            left_block_id = query_block_id - W // 2
            right_block_id = query_block_id + W // 2
            zero_key_mask = blocked_key_mask[:, -(W - (right_block_id + 1 - G)):
                                             -GB] * 0
            temp_key_mask = paddle.concat(
                [blocked_key_mask[:, GF:(right_block_id + 1)], zero_key_mask],
                axis=1)
            temp_key_mask = paddle.unsqueeze(temp_key_mask, 1)
            key_mask_list.append(temp_key_mask)
        roll_key_mask1 = paddle.concat(key_mask_list, axis=1)
        roll_key_mask1 = paddle.reshape(roll_key_mask1, [0, 0, W * bs])
        key_mask_list = []

        band_length = L - G - W // 2 * 2
        for query_block_id in range(GF + W // 2, GF + W // 2 + W):
            left_block_id = query_block_id - W // 2
            right_block_id = query_block_id + W // 2
            key_mask_list.append(blocked_key_mask[:, left_block_id:left_block_id
                                                  + band_length])
        window_key_mask = paddle.concat(key_mask_list, axis=2)
        window_key_mask = paddle.reshape(window_key_mask, [0, 0, W * bs])

        key_mask_list = []
        for query_block_id in range((L - GB) - W // 2, L - GB):
            left_block_id = query_block_id - W // 2
            right_block_id = query_block_id + W // 2
            zero_key_mask = blocked_key_mask[:, GF:GF + W - (L - left_block_id -
                                                             GB)] * 0
            temp_key_mask = paddle.concat(
                [zero_key_mask, blocked_key_mask[:, left_block_id:-GB]], axis=1)
            temp_key_mask = paddle.unsqueeze(temp_key_mask, 1)
            key_mask_list.append(temp_key_mask)
        roll_key_mask2 = paddle.concat(key_mask_list, axis=1)
        roll_key_mask2 = paddle.reshape(roll_key_mask2, [0, 0, W * bs])

        window_key_mask = paddle.concat(
            [roll_key_mask1, window_key_mask, roll_key_mask2], axis=1)
        window_key_mask = paddle.unsqueeze(window_key_mask, axis=2)
        # [B, L-G, bs, 1] * [B, L-G, 1, W*bs] -> [B, L-G, bs, W*bs]
        window_block_mask = paddle.matmul(temp_query_mask, window_key_mask)

        band_mask = paddle.concat(
            [
                global_block_mask_front, window_block_mask,
                global_block_mask_back
            ],
            axis=3)
        band_mask = paddle.unsqueeze(band_mask, 1)  # for head
        band_mask = paddle.expand(band_mask, [B, H, L - G, bs, -1])
        return band_mask

    def _get_band_matrix(self, blocked_matrix, B, T):
        '''
        return global and window matrix: [B, H, L-G, (G+W) * bs, -1]
        '''
        # blocked_matrix: [B, H, L, bs, -1]
        GB = self.num_global_blocks_back
        GF = self.num_global_blocks_front
        G = self.num_global_blocks
        R = self.num_rand_blocks
        W = self.window_size
        bs = self.block_size
        L = T // bs  # blocked length
        H = self.num_heads

        # get roll matrix
        blocked_list = []
        for query_block_id in range(GF, GF + W // 2):
            left_block_id = query_block_id - W // 2
            right_block_id = query_block_id + W // 2
            temp_blocked_matrix_list = [
                blocked_matrix[:, :, 0:(right_block_id + 1)],
                blocked_matrix[:, :, -(G + W - right_block_id - 1):]
            ]
            temp_blocked_matrix = paddle.concat(
                temp_blocked_matrix_list, axis=2)
            temp_blocked_matrix = paddle.unsqueeze(temp_blocked_matrix, axis=2)
            blocked_list.append(temp_blocked_matrix)

        # get window matrix
        band_length = L - G - W // 2 * 2
        band_matrix_list = []
        for query_block_id in range(GF + W // 2, GF + W // 2 + W):
            left_block_id = query_block_id - W // 2
            right_block_id = query_block_id + W // 2
            band_matrix_list.append(
                paddle.unsqueeze(
                    blocked_matrix[:, :, left_block_id:left_block_id +
                                   band_length],
                    axis=3))
        band_matrix = paddle.concat(band_matrix_list, axis=3)

        global_blocked_front_matrix = paddle.unsqueeze(
            blocked_matrix[:, :, :GF], axis=2)
        global_blocked_front_matrix = paddle.expand(
            global_blocked_front_matrix, [B, H, band_length, GF, bs, -1])
        global_blocked_back_matrix = paddle.unsqueeze(
            blocked_matrix[:, :, -GB:], axis=2)
        global_blocked_back_matrix = paddle.expand(
            global_blocked_back_matrix, [B, H, band_length, GB, bs, -1])
        band_matrix = paddle.concat(
            [
                global_blocked_front_matrix, band_matrix,
                global_blocked_back_matrix
            ],
            axis=3)
        blocked_list.append(band_matrix)

        for query_block_id in range(L - GB - W // 2, L - GB):
            left_block_id = query_block_id - W // 2
            right_block_id = query_block_id + W // 2
            temp_blocked_matrix_list = [
                blocked_matrix[:, :, 0:G + W - (L - left_block_id)],
                blocked_matrix[:, :, left_block_id:]
            ]
            temp_blocked_matrix = paddle.concat(
                temp_blocked_matrix_list, axis=2)
            temp_blocked_matrix = paddle.unsqueeze(temp_blocked_matrix, axis=2)
            blocked_list.append(temp_blocked_matrix)

        band_matrix = paddle.concat(blocked_list, axis=2)
        band_matrix = paddle.reshape(band_matrix,
                                     [B, H, L - G, (G + W) * bs, -1])
        return band_matrix

    def _get_rand_mask(self, blocked_query_mask, blocked_key_mask,
                       rand_mask_idx, batch_size, sequence_length):
        '''
        return random mask: [B, H, L-G, bs, R * bs]
        '''
        # rand_mask_idx: [H, T]
        # blocked_query_mask: [B, L, bs]
        # blocked_key_mask: [B, L, bs]
        bs = self.block_size
        B = batch_size
        L = sequence_length // bs
        H = self.num_heads
        G = self.num_global_blocks
        GB = self.num_global_blocks_back
        GF = self.num_global_blocks_front
        R = self.num_rand_blocks
        temp_block_key_mask = paddle.unsqueeze(blocked_key_mask, 1)
        temp_block_key_mask = paddle.expand(temp_block_key_mask, [B, H, L, -1])
        temp_block_key_mask_list = [
            paddle.gather_nd(temp_block_key_mask[b], rand_mask_idx)
            for b in range(B)
        ]
        temp_block_key_mask = paddle.concat(temp_block_key_mask_list, 0)
        temp_block_key_mask = paddle.reshape(temp_block_key_mask,
                                             [B, H, L - G, 1, R * bs])

        temp_blocked_query_mask = paddle.unsqueeze(
            blocked_query_mask[:, GF:-GB], 1)
        temp_blocked_query_mask = paddle.expand(temp_blocked_query_mask,
                                                [B, H, L - G, -1])
        temp_blocked_query_mask = paddle.reshape(temp_blocked_query_mask,
                                                 [B, H, L - G, bs, 1])

        rand_mask = paddle.matmul(temp_blocked_query_mask, temp_block_key_mask)
        return rand_mask

    def _gather_random_key_value(self, blocked_matrix, rand_mask_idx, B, T):
        '''
        return random key matrix: [B, H, L-G, R * bs, -1]
        '''
        # blocked_matrix: [B, H, L, bs, -1]
        # rand_mask_idx: [H, T]
        G = self.num_global_blocks
        H = self.num_heads
        bs = self.block_size
        L = T // bs
        R = self.num_rand_blocks
        gathered_matrix = paddle.concat(
            [
                paddle.gather_nd(blocked_matrix[b, :], rand_mask_idx)
                for b in range(B)
            ],
            axis=0)
        gathered_matrix = paddle.reshape(gathered_matrix,
                                         [B, H, L - G, R * bs, -1])
        return gathered_matrix

    def _get_global_out(self,
                        query_matrix,
                        key_matrix,
                        value_matrix,
                        key_mask,
                        d_head,
                        dropout,
                        is_front=True):
        GB = self.num_global_blocks_back
        GF = self.num_global_blocks_front
        if is_front:
            global_query_matrix = query_matrix[:, :, 0:GF * self.block_size]
        else:
            global_query_matrix = query_matrix[:, :, -GB * self.block_size:]
        global_product = paddle.matmul(
            global_query_matrix, key_matrix, transpose_y=True)
        global_product = global_product * (d_head**-0.5)
        global_product += (1 - key_mask) * -1e6
        global_weights = F.softmax(global_product)
        if dropout:
            global_weights = F.dropout(
                global_weights,
                dropout,
                training=self.training,
                mode="upscale_in_train")
        # [B, H, GF*bs, T] * [B, H, T, D] -> [B, H, GF*bs, D]
        global_product = paddle.matmul(global_weights, value_matrix)
        return global_product

    def forward(self,
                query_matrix,
                key_matrix,
                value_matrix,
                d_head,
                attn_mask=None,
                rand_mask_idx=None,
                query_mask=None,
                key_mask=None,
                dropout=None):
        '''
            query_matrix: [B, H, T, D]
            key_matrix: [B, H, T, D]
            value_matrix: [B, H, T, D]
            query_mask: [B, 1, T, 1]  bool mask
            key_mask: [B, 1, 1, T]    bool mask
            rand_mask_idx: [H, T//bs, bs]
            Global Attention
            Random Attention
            Window Attention
            key_matrix分为4块：
            
        '''
        B = query_matrix.shape[0]  # batch_size
        H = self.num_heads
        T = query_matrix.shape[2]  # sequence_length
        D = query_matrix.shape[3]  # size per head
        G = self.num_global_blocks
        GB = self.num_global_blocks_back
        GF = self.num_global_blocks_front
        R = self.num_rand_blocks
        W = self.window_size
        bs = self.block_size
        L = T // bs  # blocked length

        blocked_query_matrix = paddle.reshape(query_matrix, [B, H, L, bs, -1])
        blocked_key_matrix = paddle.reshape(key_matrix, [B, H, L, bs, -1])
        blocked_value_matrix = paddle.reshape(value_matrix, [B, H, L, bs, -1])
        blocked_query_mask = paddle.reshape(query_mask, [B, L, bs])
        blocked_key_mask = paddle.reshape(key_mask, [B, L, bs])

        # 所有global_block中的query与所有key做点积
        ########################## global_front_product ##########################
        global_front_out = self._get_global_out(
            query_matrix, key_matrix, value_matrix, key_mask, d_head, dropout)

        ########################## global_back_product ##########################
        global_back_out = self._get_global_out(query_matrix, key_matrix,
                                               value_matrix, key_mask, d_head,
                                               dropout, False)

        ########################### second_product #############################

        # create second matrix
        # [B, 1, L-G, bs, (G+W)*bs]
        band_mask = self._get_band_mask(blocked_query_mask, blocked_key_mask, B,
                                        T)
        # [B, H, L-G, bs, R*bs]
        # rand_mask = self._get_rand_mask(blocked_query_mask, blocked_key_mask,
        #                                 rand_mask_idx, B, T)
        # [B, H, L-G, bs, (G+W+R)*bs]
        # second_mask = paddle.concat([band_mask, rand_mask], axis=4)
        second_mask = band_mask

        # [B, H, L-G, R, bs, -1]
        # random_keys = self._gather_random_key_value(blocked_key_matrix,
        #                                             rand_mask_idx, B, T)
        # random_values = self._gather_random_key_value(blocked_value_matrix,
        #                                               rand_mask_idx, B, T)

        band_keys_matrix = self._get_band_matrix(blocked_key_matrix, B, T)
        band_value_matrix = self._get_band_matrix(blocked_value_matrix, B, T)

        # [B, H, L - G, bs, -1]
        second_query_matrix = blocked_query_matrix[:, :, GF:-GB]
        # [B, H, L - G, (G+W+R)*bs, -1]
        # second_key_matrix = paddle.concat(
        #     [band_keys_matrix, random_keys], axis=3)
        second_key_matrix = band_keys_matrix
        # [B, H, L - G, (G+W+R)*bs, -1]
        # second_value_matrix = paddle.concat(
        #     [band_value_matrix, random_values], axis=3)
        second_value_matrix = band_value_matrix

        second_product = paddle.matmul(
            second_query_matrix, second_key_matrix, transpose_y=True)
        second_product = second_product * (d_head**-0.5)
        second_product += (1 - second_mask) * -1e6
        second_weights = F.softmax(second_product)
        if dropout:
            second_weights = F.dropout(
                second_weights,
                dropout,
                training=self.training,
                mode="upscale_in_train")
        # [B, H, L - G, bs, (G+W+R)*bs] *  [B, H, L - G, (G+W+R)*bs, -1] = [B, H, L-G, bs, -1]
        second_out = paddle.matmul(second_weights, second_value_matrix)
        second_out = paddle.reshape(second_out, [B, H, (L - G) * bs, -1])

        # [B, H, T, D]
        out = paddle.concat(
            [global_front_out, second_out, global_back_out], axis=2)
        out = out * query_mask
        return out


class MultiHeadAttention(Layer):

    Cache = collections.namedtuple("Cache", ["k", "v"])
    StaticCache = collections.namedtuple("StaticCache", ["k", "v"])

    def __init__(self,
                 embed_dim,
                 num_heads,
                 dropout=0.,
                 kdim=None,
                 vdim=None,
                 weight_attr=None,
                 bias_attr=None,
                 block_size=1,
                 window_size=3,
                 num_global_blocks=1,
                 num_rand_blocks=1,
                 seed=None,
                 attention_type="bigbird"):

        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.num_heads = num_heads
        self.dropout = dropout

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = Linear3D(
            embed_dim,
            num_heads,
            self.head_dim,
            weight_attr,
            bias_attr=bias_attr)
        self.k_proj = Linear3D(
            embed_dim,
            num_heads,
            self.head_dim,
            weight_attr,
            bias_attr=bias_attr)
        self.v_proj = Linear3D(
            embed_dim,
            num_heads,
            self.head_dim,
            weight_attr,
            bias_attr=bias_attr)
        self.out_proj = nn.Linear(
            embed_dim, embed_dim, weight_attr, bias_attr=bias_attr)

        self.attn_impl = AttentionRegistry.cls_dict[attention_type](
            num_heads, block_size, window_size, num_global_blocks,
            num_rand_blocks, seed)

    def _prepare_qkv(self, query, key, value, cache=None):
        q = self.q_proj(query)

        if isinstance(cache, self.StaticCache):
            # for encoder-decoder attention in inference and has cached
            k, v = cache.k, cache.v
        else:
            k, v = self.compute_kv(key, value)

        if isinstance(cache, self.Cache):
            # for decoder self-attention in inference
            k = paddle.concat([cache.k, k], axis=2)
            v = paddle.concat([cache.v, v], axis=2)
            cache = self.Cache(k, v)

        return (q, k, v) if cache is None else (q, k, v, cache)

    def compute_kv(self, key, value):
        k = self.k_proj(key)
        v = self.v_proj(value)
        return k, v

    def gen_cache(self, key, value=None, type=Cache):
        if type == MultiHeadAttention.StaticCache:  # static_kv
            k, v = self.compute_kv(key, value)
            return self.StaticCache(k, v)
        elif value is None:  # incremental_state
            k = paddle.full(
                shape=[-1, self.num_heads, 0, self.head_dim],
                fill_value=0,
                dtype=key.dtype)

            v = paddle.full(
                shape=[-1, self.num_heads, 0, self.head_dim],
                fill_value=0,
                dtype=key.dtype)
            return self.Cache(k, v)
        else:
            # incremental_state with initial value, mainly for usage like UniLM
            return self.Cache(key, value)

    def forward(self,
                query,
                key,
                value,
                attn_mask=None,
                rand_mask_idx=None,
                query_mask=None,
                key_mask=None,
                cache=None):
        key = query if key is None else key
        value = query if value is None else value
        # compute q ,k ,v
        if cache is None:
            q, k, v = self._prepare_qkv(query, key, value, cache)
        else:
            q, k, v, cache = self._prepare_qkv(query, key, value, cache)

        out = self.attn_impl(q, k, v, self.head_dim, attn_mask, rand_mask_idx,
                             query_mask, key_mask, self.dropout)
        # combine heads
        out = paddle.transpose(out, perm=[0, 2, 1, 3])
        out = paddle.reshape(x=out, shape=[0, 0, out.shape[2] * out.shape[3]])

        # project to output
        out = self.out_proj(out)

        outs = [out]
        if cache is not None:
            outs.append(cache)
        return out if len(outs) == 1 else tuple(outs)