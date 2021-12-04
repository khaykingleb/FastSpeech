import torch
from torch.nn.utils.rnn import pad_sequence

from tts.collate_fn import Batch
from tts.spectrogram import MelSpectrogram
from tts.aligner import GraphemeAligner


def prepare_batch(
    batch: Batch, 
    melspectrogramer: MelSpectrogram, 
    aligner: GraphemeAligner, 
    device
) -> Batch:

    batch.melspec = melspectrogramer(batch.waveform.to(device))

    durations = aligner(batch.waveform, batch.waveform_length, batch.transcript)

    durations_melspec = []
    for index in range(batch.waveform.shape[0]):
        durations_wave = durations[index]
        durations_normalized = durations_wave / durations_wave.sum()

        melspec = melspectrogramer(batch.waveform[index][:batch.waveform_length[index]])
        durations_melspec.append(torch.round(durations_normalized * melspec.shape[1]).int())
        
    batch.durations = pad_sequence(durations_melspec).permute(1, 0)

    return batch.to(device)