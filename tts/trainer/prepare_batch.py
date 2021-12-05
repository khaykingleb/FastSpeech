from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch

from tts.collate_fn import Batch


def prepare_batch(
    batch: Batch, 
    melspectrogramer: nn.Module, 
    aligner: nn.Module, 
    device: torch.device
) -> Batch:

    batch.melspec = melspectrogramer(batch.waveform.to(device))

    durations = aligner(
        batch.waveform.to(device), 
        batch.waveform_length.to(device), 
        batch.transcript
    )

    durations_melspec = []
    for index in range(batch.waveform.shape[0]):
        durations_wave = durations[index]
        durations_normalized = durations_wave / durations_wave.sum()

        melspec = melspectrogramer(
            batch.waveform[index][:batch.waveform_length[index]].to(device)
        )
        durations_melspec.append(
            torch.round(durations_normalized * melspec.shape[1]).float()
        )
        
    batch.durations = pad_sequence(durations_melspec).permute(1, 0)

    return batch.to(device)