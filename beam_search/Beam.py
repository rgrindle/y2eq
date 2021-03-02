"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Feb 24, 2021

PURPOSE: Implement beam search

NOTES:

TODO:
"""
from srvgd.utils.eval import get_string, default_map, normalize

import torch
from scipy.special import softmax

import copy


def get_encoder_output(model, encoder_input):

    with torch.no_grad():
        encoder_conved, encoder_combined = model.encoder(encoder_input)
        return encoder_conved, encoder_combined


def beam_search(beam_size, encoder_input, model,
                device=torch.device('cpu'),
                max_len=100):

    if type(encoder_input) != torch.Tensor:
        encoder_input = torch.Tensor(encoder_input).to(device)
    else:
        encoder_input = encoder_input.to(device)

    encoder_input = encoder_input.unsqueeze(0)

    model.eval()
    encoder_states = get_encoder_output(model, encoder_input)

    mapping = default_map()
    prev_indices = [mapping['START']]

    beam_list = [Beam(beam_size=beam_size,
                      model=model,
                      sequence=prev_indices,
                      encoder_states=encoder_states)]

    while not np.all([beam.is_done() for beam in beam_list]):
        new_beams_list = []
        for beam in beam_list:
            if beam.is_done():
                new_beams_list.append(beam)
                continue

            best_outputs, best_indices = beam.eval_decoder()

            for index, output in zip(best_indices, best_outputs):
                new_beam = copy.deepcopy(beam)
                new_beam.sequence.append(index)
                new_beam.update_prob(output)
                new_beams_list.append(new_beam)

        log_prob_list = [n.log_prob for n in new_beams_list]
        beam_list = [new_beams_list[i] for i in np.argsort(log_prob_list)[:beam_size]]

    return [get_string(beam.sequence) for beam in beam_list]


class Beam:

    def __init__(self, beam_size, model, encoder_states,
                 sequence, device=torch.device('cpu'),
                 max_sequence_length=67):
        self.beam_size = beam_size
        self.model = model
        self.device = device
        self.encoder_states = encoder_states
        self.sequence = sequence
        self.log_prob = 0.0
        self.max_sequence_length = max_sequence_length

    def eval_decoder(self):
        prev_tensor = torch.LongTensor(self.sequence).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output, attention = model.decoder(prev_tensor, *self.encoder_states)
            output = output.numpy()[0, -1]
            output = softmax(output)
            sort_indices = np.argsort(output)[::-1]
            best_indices = sort_indices[:self.beam_size]
            best_outputs = output[best_indices]
            return best_outputs, best_indices

    def update_prob(self, p):
        assert 0 <= p <= 1
        self.log_prob += -np.log(p)

    def is_done(self):
        mapping = default_map()
        return mapping['END'] in self.sequence or \
            len(self.sequence) >= self.max_sequence_length


if __name__ == '__main__':
    from srvgd.architecture.torch.get_model import get_model
    import numpy as np

    # get model
    model = get_model(torch.device('cpu'),
                      path='../models/',
                      load_weights='cnn_dataset_train_ff1000_batchsize2000_lr0.0001_clip1_layers10_900.pt')

    # get input
    x = np.arange(0.1, 3.1, 0.1)[:, None]
    y = x**4 + x
    y = normalize(y)

    pred_outputs = beam_search(beam_size=10,
                               encoder_input=torch.Tensor(y),
                               model=model)

    print('equations found by beam search')
    for p in pred_outputs:
        print(p)
