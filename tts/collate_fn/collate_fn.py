from typing import Tuple, Dict, Optional, List, Union
from dataclasses import dataclass

from torch.nn.utils.rnn import pad_sequence
import torch


@dataclass
class Batch:
    waveform: torch.Tensor
    waveform_length: torch.Tensor
    transcript: List[str]
    tokens: torch.Tensor
    token_lengths: torch.Tensor
    durations: Optional[torch.Tensor] = None
    mel_specs: Optional[torch.Tensor] = None
        
    def to(self, device: torch.device) -> 'Batch':
        batch = Batch(
            self.waveform.to(device),
            self.waveform_length.to(device),
            self.transcript,
            self.tokens.to(device),
            self.token_lengths.to(device),
            self.durations.to(device),
            self.mel_specs.to(device)
        )
        return batch

class LJSpeechCollator:

    def __call__(self, instances: List[Tuple]) -> Dict:
        waveform, waveform_length, transcript, tokens, token_lengths = list(
            zip(*instances)
        )

        waveform = pad_sequence([
            waveform_[0] for waveform_ in waveform
        ]).transpose(0, 1)
        waveform_length = torch.cat(waveform_length)

        tokens = pad_sequence([
            tokens_[0] for tokens_ in tokens
        ]).transpose(0, 1)
        token_lengths = torch.cat(token_lengths)

        batch = Batch(waveform, waveform_length, transcript, tokens, token_lengths)
        return batch
