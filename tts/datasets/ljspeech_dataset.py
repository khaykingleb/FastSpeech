import torchaudio
import torch

import string

class LJSpeechDataset(torchaudio.datasets.LJSPEECH):

    def __init__(self, root):
        super().__init__(root=root)
        self.tokenizer = torchaudio.pipelines.TACOTRON2_GRIFFINLIM_CHAR_LJSPEECH.get_text_processor()

    def __getitem__(self, index: int):
        waveform, _, _, transcript = super().__getitem__(index)
        waveforn_length = torch.tensor([waveform.shape[-1]]).int()

        transcript = transcript.encode("ascii", "ignore").decode() \
            .translate(str.maketrans('', '', string.punctuation)) \
            .strip()
        
        tokens, token_lengths = self.tokenizer(transcript)
        
        return waveform, waveforn_length, transcript, tokens, token_lengths
    
    def decode(self, tokens, lengths):
        result = []
        for tokens_, length in zip(tokens, lengths):
            text = "".join([
                self.tokenizer.tokens[token]
                for token in tokens_[:length]
            ])
            result.append(text)

        return result
                