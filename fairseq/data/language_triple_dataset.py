# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import numpy as np
import torch

from fairseq import utils

from . import data_utils, FairseqDataset


def collate(
        samples, pad_idx, eos_idx, left_pad_source=True, left_pad_target=False,
        input_feeding=True,
):
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx, eos_idx, left_pad, move_eos_to_beginning,
        )

    id = torch.LongTensor([s['id'] for s in samples])
    src_tokens = merge('source', left_pad=left_pad_source)
    # sort by descending source length
    src_lengths = torch.LongTensor([s['source'].numel() for s in samples])
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    src_tokens = src_tokens.index_select(0, sort_order)

    prev_clean_output_tokens = None
    prev_trans_output_tokens = None
    target_clean = None
    target_trans = None
    tgt_clean_mask = None
    if samples[0].get('target_clean', None) is not None:
        target_clean = merge('target_clean', left_pad=left_pad_target)
        target_clean = target_clean.index_select(0, sort_order)
        ntokens = sum(len(s['target_clean']) for s in samples)
        tgt_clean_mask = target_clean.eq(pad_idx)

        if input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_clean_output_tokens = merge(
                'target_clean',
                left_pad=left_pad_target,
                move_eos_to_beginning=True,
            )
            prev_clean_output_tokens = prev_clean_output_tokens.index_select(0, sort_order)
    if samples[0].get('target_trans', None) is not None:
        target_trans = merge('target_trans', left_pad=left_pad_target)
        target_trans = target_trans.index_select(0, sort_order)
        ntokens = sum(len(s['target_trans']) for s in samples)  # TODO: check this

        if input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_trans_output_tokens = merge(
                'target_trans',
                left_pad=left_pad_target,
                move_eos_to_beginning=True,
            )
            prev_trans_output_tokens = prev_trans_output_tokens.index_select(0, sort_order)
    else:
        ntokens = sum(len(s['source']) for s in samples)

    batch = {
        'id': id,
        'nsentences': len(samples),
        'ntokens': ntokens,
        'net_input': {
            'src_tokens': src_tokens,
            'src_lengths': src_lengths,
        },
        'target_clean': target_clean,
        'target_trans': target_trans,
    }
    if prev_clean_output_tokens is not None:
        batch['net_input']['prev_clean_output_tokens'] = prev_clean_output_tokens
        batch['net_input']['tgt_clean_mask'] = tgt_clean_mask
    if prev_trans_output_tokens is not None:
        batch['net_input']['prev_trans_output_tokens'] = prev_trans_output_tokens
    return batch


def generate_dummy_batch(num_tokens, collate_fn, src_dict, src_len=128, tgt_clean_dict=None,
                         tgt_trans_dict=None, tgt_len=128):
    """Return a dummy batch with a given number of tokens."""
    bsz = num_tokens // max(src_len, tgt_len)
    return collate_fn([
        {
            'id': i,
            'source': src_dict.dummy_sentence(src_len),
            'target_clean': tgt_clean_dict.dummy_sentence(tgt_len) if tgt_clean_dict is not None else None,
            'target_trans': tgt_trans_dict.dummy_sentence(tgt_len) if tgt_trans_dict is not None else None,
        }
        for i in range(bsz)
    ])


class LanguageTripleDataset(FairseqDataset):
    """
    A pair of torch.utils.data.Datasets.

    Args:
        src (torch.utils.data.Dataset): source dataset to wrap
        src_sizes (List[int]): source sentence lengths
        src_dict (~fairseq.data.Dictionary): source vocabulary
        tgt (torch.utils.data.Dataset, optional): target dataset to wrap
        tgt_sizes (List[int], optional): target sentence lengths
        tgt_dict (~fairseq.data.Dictionary, optional): target vocabulary
        left_pad_source (bool, optional): pad source tensors on the left side
            (default: True).
        left_pad_target (bool, optional): pad target tensors on the left side
            (default: False).
        max_source_positions (int, optional): max number of tokens in the
            source sentence (default: 1024).
        max_target_positions (int, optional): max number of tokens in the
            target sentence (default: 1024).
        shuffle (bool, optional): shuffle dataset elements before batching
            (default: True).
        input_feeding (bool, optional): create a shifted version of the targets
            to be passed into the model for input feeding/teacher forcing
            (default: True).
        remove_eos_from_source (bool, optional): if set, removes eos from end
            of source if it's present (default: False).
        append_eos_to_target (bool, optional): if set, appends eos to end of
            target if it's absent (default: False).
    """

    def __init__(
            self, src, src_sizes, src_dict,
            tgt_clean=None, tgt_clean_sizes=None, tgt_clean_dict=None,
            tgt_trans=None, tgt_trans_sizes=None, tgt_trans_dict=None,
            left_pad_source=True, left_pad_target=False,
            max_source_positions=1024, max_target_positions=1024,
            shuffle=True, input_feeding=True, remove_eos_from_source=False, append_eos_to_target=False,
    ):
        if tgt_trans_dict is not None:
            assert src_dict.pad() == tgt_trans_dict.pad()
            assert src_dict.pad() == tgt_clean_dict.pad()
            assert src_dict.eos() == tgt_trans_dict.eos()
            assert src_dict.eos() == tgt_clean_dict.eos()
            assert src_dict.unk() == tgt_trans_dict.unk()
            assert src_dict.unk() == tgt_clean_dict.unk()
        self.src = src
        self.tgt_clean = tgt_clean
        self.tgt_trans = tgt_trans
        self.src_sizes = np.array(src_sizes)
        self.tgt_clean_sizes = np.array(tgt_clean_sizes) if tgt_clean_sizes is not None else None
        self.tgt_trans_sizes = np.array(tgt_trans_sizes) if tgt_trans_sizes is not None else None
        self.src_dict = src_dict
        self.tgt_clean_dict = tgt_clean_dict
        self.tgt_trans_dict = tgt_trans_dict
        self.left_pad_source = left_pad_source
        self.left_pad_target = left_pad_target
        self.max_source_positions = max_source_positions
        self.max_target_positions = max_target_positions
        self.shuffle = shuffle
        self.input_feeding = input_feeding
        self.remove_eos_from_source = remove_eos_from_source
        self.append_eos_to_target = append_eos_to_target

    def __getitem__(self, index):
        tgt_clean_item = self.tgt_clean[index] if self.tgt_clean is not None else None
        tgt_trans_item = self.tgt_trans[index] if self.tgt_trans is not None else None
        src_item = self.src[index]
        # Append EOS to end of tgt sentence if it does not have an EOS and remove
        # EOS from end of src sentence if it exists. This is useful when we use
        # use existing datasets for opposite directions i.e., when we want to
        # use tgt_dataset as src_dataset and vice versa
        if self.append_eos_to_target:
            raise NotImplementedError
            eos = self.tgt_dict.eos() if self.tgt_dict else self.src_dict.eos()
            if self.tgt and self.tgt[index][-1] != eos:
                tgt_item = torch.cat([self.tgt[index], torch.LongTensor([eos])])

        if self.remove_eos_from_source:
            raise NotImplementedError
            eos = self.src_dict.eos()
            if self.src[index][-1] == eos:
                src_item = self.src[index][:-1]

        return {
            'id': index,
            'source': src_item,
            'target_clean': tgt_clean_item,
            'target_trans': tgt_trans_item,
        }

    def __len__(self):
        return len(self.src)

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate

        Returns:
            dict: a mini-batch with the following keys:

                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:

                  - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the source sentence of shape `(bsz, src_len)`. Padding will
                    appear on the left if *left_pad_source* is ``True``.
                  - `src_lengths` (LongTensor): 1D Tensor of the unpadded
                    lengths of each source sentence of shape `(bsz)`
                  - `prev_output_tokens` (LongTensor): a padded 2D Tensor of
                    tokens in the target sentence, shifted right by one position
                    for input feeding/teacher forcing, of shape `(bsz,
                    tgt_len)`. This key will not be present if *input_feeding*
                    is ``False``. Padding will appear on the left if
                    *left_pad_target* is ``True``.

                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the left if *left_pad_target* is ``True``.
        """
        return collate(
            samples, pad_idx=self.src_dict.pad(), eos_idx=self.src_dict.eos(),
            left_pad_source=self.left_pad_source, left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding,
        )

    def get_dummy_batch(self, num_tokens, max_positions, src_len=128, tgt_len=128):
        """Return a dummy batch with a given number of tokens."""
        src_len, tgt_len = utils.resolve_max_positions(
            (src_len, tgt_len),
            max_positions,
            (self.max_source_positions, self.max_target_positions),
        )
        return generate_dummy_batch(num_tokens, self.collater, self.src_dict, src_len,
                                    self.tgt_clean_dict, self.tgt_trans_dict, tgt_len)

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return max(self.src_sizes[index], self.tgt_clean_sizes[index] if self.tgt_clean_sizes is not None else 0,
                   self.tgt_trans_sizes[index] if self.tgt_trans_sizes is not None else 0)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return (self.src_sizes[index], self.tgt_clean_sizes[index] if self.tgt_clean_sizes is not None else 0,
                self.tgt_trans_sizes[index] if self.tgt_trans_sizes is not None else 0)

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self))
        else:
            indices = np.arange(len(self))
        if self.tgt_clean_sizes is not None:  # TODO: check this
            indices = indices[np.argsort(self.tgt_clean_sizes[indices], kind='mergesort')]
        if self.tgt_trans_sizes is not None:
            indices = indices[np.argsort(self.tgt_trans_sizes[indices], kind='mergesort')]
        return indices[np.argsort(self.src_sizes[indices], kind='mergesort')]

    @property
    def supports_prefetch(self):
        return (
                getattr(self.src, 'supports_prefetch', False)
                and (getattr(self.tgt_clean, 'supports_prefetch', False) or self.tgt_clean is None)
                and (getattr(self.tgt_trans, 'supports_prefetch', False) or self.tgt_trans is None)
        )

    def prefetch(self, indices):
        self.src.prefetch(indices)
        if self.tgt_clean is not None:
            self.tgt_clean.prefetch(indices)
        if self.tgt_trans is not None:
            self.tgt_trans.prefetch(indices)
