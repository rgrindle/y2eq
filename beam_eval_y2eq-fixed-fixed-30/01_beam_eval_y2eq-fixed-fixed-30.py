from srvgd.eval_scripts.get_ff import get_ff_beam_search
from srvgd.architecture.torch.get_model import get_model

import torch

import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
file_endname = '_dataset_train_ff1000_batchsize32_lr0.0001_clip1_layers10_105'
model = get_model(device,
                  path='../models/',
                  load_weights='cnn{}.pt'.format(file_endname))

with open('00_y_int_normalized_list.json', 'r') as json_file:
    y_int_list = json.load(json_file)


get_ff_beam_search(y_int_list=y_int_list,
                   model=model,
                   beam_size=3)
