from typing import List, NamedTuple, DefaultDict

from torch import Tensor
from typing import List, Union
import numpy as np
from .char_text_encoder import CharTextEncoder


class Hypothesis(NamedTuple):
    text: str
    prob: float


class PhonemeTextEncoder(CharTextEncoder):
    EMPTY_TOKS = ["sp", "spn", "sil"]

    def __init__(self, alphabet: List[str] = None):
        alphabet = [
            "AA",
            "AA0",
            "AA1",
            "AA2",
            "AE",
            "AE0",
            "AE1",
            "AE2",
            "AH",
            "AH0",
            "AH1",
            "AH2",
            "AO",
            "AO0",
            "AO1",
            "AO2",
            "AW",
            "AW0",
            "AW1",
            "AW2",
            "AY",
            "AY0",
            "AY1",
            "AY2",
            "B",
            "CH",
            "D",
            "DH",
            "EH",
            "EH0",
            "EH1",
            "EH2",
            "ER",
            "ER0",
            "ER1",
            "ER2",
            "EY",
            "EY0",
            "EY1",
            "EY2",
            "F",
            "G",
            "HH",
            "IH",
            "IH0",
            "IH1",
            "IH2",
            "IY",
            "IY0",
            "IY1",
            "IY2",
            "JH",
            "K",
            "L",
            "M",
            "N",
            "NG",
            "OW",
            "OW0",
            "OW1",
            "OW2",
            "OY",
            "OY0",
            "OY1",
            "OY2",
            "P",
            "R",
            "S",
            "SH",
            "T",
            "TH",
            "UH",
            "UH0",
            "UH1",
            "UH2",
            "UW",
            "UW0",
            "UW1",
            "UW2",
            "V",
            "W",
            "Y",
            "Z",
            "ZH"
        ]
        super().__init__(alphabet)
        vocab = self.EMPTY_TOKS + list(self.alphabet)
        self.ind2ph = dict(enumerate(vocab))
        self.ph2ind = {v: k for k, v in self.ind2ph.items()}
    def encode(self, text: str) -> List[int]:
        phones = text.split(' ')
        try:
            return Tensor([self.ph2ind[ph] for ph in phones]).unsqueeze(0)
        except KeyError as e:
            unknown_chars = set([ph for ph in phones if ph not in self.ph2ind])
            raise Exception(
                f"Can't encode text '{text}'. Unknown chars: '{' '.join(unknown_chars)}'")

    def decode(self, vector: Union[Tensor, np.ndarray, List[int]]):
        return ' '.join([self.ind2ph[int(ind)] for ind in vector]).strip()