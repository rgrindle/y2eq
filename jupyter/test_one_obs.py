"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Jan 14, 2020

PURPOSE: Get output of trained NN on a single equation.

NOTES:

TODO:
"""
from get_model import get_model
from tensor_dataset import TensorDatasetCPU as TensorDataset  # noqa: F401
from eqlearner.dataset.processing.tokenization import get_string
import torch
import numpy as np


# file_endname = '_layers10_clip1_dropoutTrue_lr1e-4_no_duplicates'
# file_endname = '_epochs100_0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = get_model(device, 'cnn.pt')
model.eval()

# test_data = torch.load('dataset_test.pt', map_location=device)
# test_data = DataLoader(test_data, batch_size=1000)
# output_data = []
with torch.no_grad():
    # for batch in test_data:
    #     print('batch')
    #     src = batch[0]
    #     trg = batch[1]
    src = torch.Tensor([[0.0352, 0.1318, 0.2650, 0.4073, 0.5397, 0.6567, 0.7612, 0.8559, 0.9358,
                         0.9882, 1.0000, 0.9674, 0.9002, 0.8174, 0.7373, 0.6668, 0.5979, 0.5147,
                         0.4055, 0.2742, 0.1424, 0.0418, 0.0000, 0.0292, 0.1223, 0.2591, 0.4166,
                         0.5773, 0.7304, 0.8676],
                        [0.0352, 0.1318, 0.2650, 0.4073, 0.5397, 0.6567, 0.7612, 0.8559, 0.9358,
                         0.9882, 1.0000, 0.9674, 0.9002, 0.8174, 0.7373, 0.6668, 0.5979, 0.5147,
                         0.4055, 0.2742, 0.1424, 0.0418, 0.0000, 0.0292, 0.1223, 0.2591, 0.4166,
                         0.5773, 0.7304, 0.8676]])
    print(src)
    trg = torch.Tensor([[12, 2, 5, 1, 6, 7, 21, 9, 2, 5, 1, 6, 7, 17, 13, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [12, 2, 5, 1, 6, 7, 21, 9, 2, 5, 1, 6, 7, 17, 13, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]).long()
    output, _ = model(src, trg[:, :-1])
token_seq_out = np.argmax(output.cpu().numpy(), axis=2)
print(token_seq_out)
print(get_string(token_seq_out[0]))
print(get_string(token_seq_out[1]))

# ouptut equation is
# sin(x)**6+sin(x)**2END)**6)6666666666666)6666666666)66666)66)66))6666666666
