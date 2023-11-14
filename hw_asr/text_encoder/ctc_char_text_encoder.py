from typing import List, NamedTuple, DefaultDict

import torch

from .char_text_encoder import CharTextEncoder


class Hypothesis(NamedTuple):
    text: str
    prob: float


class CTCCharTextEncoder(CharTextEncoder):
    EMPTY_TOK = "^"

    def __init__(self, alphabet: List[str] = None):
        super().__init__(alphabet)
        vocab = [self.EMPTY_TOK] + list(self.alphabet)
        self.ind2char = dict(enumerate(vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

    def ctc_decode(self, inds: List[int]) -> str:
        decoded = []
        was_empty = False
        for ind in inds:
            cur_char = self.ind2char[ind]
            if cur_char == self.EMPTY_TOK:
                was_empty = True
            else:
                if was_empty or len(decoded) == 0 or decoded[-1] != cur_char:
                    decoded.append(cur_char)
                was_empty = False
        return ''.join(decoded)

    def __extend_and_merge(self, frame, state):
        new_state = DefaultDict(float)
        for next_char_index, next_char_prob in enumerate(frame):
            next_char = self.ind2char[next_char_index]
            for (pref, last_char), pref_proba in state.items():
                if next_char == last_char:
                    new_pref = pref
                else:
                    if next_char != self.EMPTY_TOK:
                        new_pref = pref + next_char
                    else:
                        new_pref = pref
                    last_char = next_char
                new_state[(new_pref, last_char)] += pref_proba * next_char_prob
        return new_state

    def __truncate(self, state, beam_size):
        state_list = list(state.items())
        state_list.sort(key=lambda x: -x[1])
        return dict(state_list[:beam_size])

    def ctc_beam_search(self, probs: torch.tensor, probs_length,
                        beam_size: int = 100) -> List[Hypothesis]:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        """
        assert len(probs.shape) == 2
        char_length, voc_size = probs.shape
        probs = probs.detach().numpy()
        assert voc_size == len(self.ind2char)
        hypos: List[Hypothesis] = []

        state = {('', self.EMPTY_TOK): 1.0}
        for frame in probs:
            state = self.__extend_and_merge(frame, state)
            state = self.__truncate(state, beam_size)
        hypos = [Hypothesis(v[0][0], v[-1]) for v in list(state.items())]
        return sorted(hypos, key=lambda x: x.prob, reverse=True)