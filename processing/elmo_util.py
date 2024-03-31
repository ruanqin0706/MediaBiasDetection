import torch
import numpy
from typing import List, Tuple, Iterable

from allennlp.common.util import lazy_groups_of
from allennlp.data.token_indexers.elmo_indexer import ELMoTokenCharactersIndexer
from allennlp.modules.elmo import _ElmoBiLm, batch_to_ids
from allennlp.nn.util import remove_sentence_boundaries


# modified from: https://github.com/allenai/allennlp/blob/v0.9.0/allennlp/commands/elmo.py
class ElmoEmbedder(object):
    def __init__(self,
                 options_file,
                 weight_file,
                 cuda_device: int = -1) -> None:
        """
        Parameters
        ----------
        options_file : ``str``, optional
            A path or URL to an ELMo options file.
        weight_file : ``str``, optional
            A path or URL to an ELMo weights file.
        cuda_device : ``int``, optional, (default=-1)
            The GPU device to run on.
        """
        self.indexer = ELMoTokenCharactersIndexer()

        self.elmo_bilm = _ElmoBiLm(options_file, weight_file)
        if cuda_device >= 0:
            self.elmo_bilm = self.elmo_bilm.cuda(device=cuda_device)

        self.cuda_device = cuda_device

    def batch_to_embeddings(self, batch: List[List[str]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        batch : ``List[List[str]]``, required
            A list of tokenized sentences.
        Returns
        -------
            A tuple of tensors, the first representing activations (batch_size, 3, num_timesteps, 1024) and
        the second a mask (batch_size, num_timesteps).
        """
        character_ids = batch_to_ids(batch)
        if self.cuda_device >= 0:
            character_ids = character_ids.cuda(device=self.cuda_device)

        bilm_output = self.elmo_bilm(character_ids)
        layer_activations = bilm_output['activations']
        mask_with_bos_eos = bilm_output['mask']

        # without_bos_eos is a 3 element list of (activation, mask) tensor pairs,
        # each with size (batch_size, num_timesteps, dim and (batch_size, num_timesteps)
        # respectively.
        without_bos_eos = [remove_sentence_boundaries(layer, mask_with_bos_eos)
                           for layer in layer_activations]
        # Converts a list of pairs (activation, mask) tensors to a single tensor of activations.
        activations = torch.cat([ele[0].unsqueeze(1) for ele in without_bos_eos], dim=1)
        # The mask is the same for each ELMo vector, so just take the first.
        mask = without_bos_eos[0][1]

        return activations, mask

    def embed_batch(self, batch: List[List[str]]) -> List[numpy.ndarray]:
        """
        Computes the ELMo embeddings for a batch of tokenized sentences.
        Please note that ELMo has internal state and will give different results for the same input.
        See the comment under the class definition.
        Parameters
        ----------
        batch : ``List[List[str]]``, required
            A list of tokenized sentences.
        Returns
        -------
            A list of tensors, each representing the ELMo vectors for the input sentence at the same index.
        """
        elmo_embeddings = []

        # todo case 0?
        embeddings, mask = self.batch_to_embeddings(batch)
        for i in range(len(batch)):
            length = int(mask[i, :].sum())
            elmo_embeddings.append(embeddings[i, :, :length, :].detach().cpu().numpy())

        return elmo_embeddings

    def embed_sentences(self,
                        sentences: Iterable[List[str]],
                        batch_size=16) -> Iterable[numpy.ndarray]:
        """
        Computes the ELMo embeddings for a iterable of sentences.
        Please note that ELMo has internal state and will give different results for the same input.
        See the comment under the class definition.
        Parameters
        ----------
        sentences : ``Iterable[List[str]]``, required
            An iterable of tokenized sentences.
        batch_size : ``int``, required
            The number of sentences ELMo should process at once.
        Returns
        -------
            A list of tensors, each representing the ELMo vectors for the input sentence at the same index.
        """
        for batch in lazy_groups_of(iter(sentences), batch_size):
            yield from self.embed_batch(batch)
