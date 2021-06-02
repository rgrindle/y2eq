"""
AUTHOR: Ryan Grindle

LAST MODIFIED: May 14, 2021

PURPOSE: Create a version of y2eq that is a
         transformer.

NOTES:

TODO:
"""
from srvgd.architecture.transformer.train_and_validate import get_dataset, split_dataset, train_many_epochs
from srvgd.architecture.transformer.y2eq_transformer import y2eqTransformer

import torch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
y2eq_trans_model = y2eqTransformer(src_input_size=4,
                                   src_max_len=30).to(device)

if __name__ == '__main__':
    # Get number of trainable parameters
    num_params = sum(p.numel() for p in y2eq_trans_model.parameters() if p.requires_grad)
    print('Num trainable params:', num_params)

    # Get dataset
    dataset_name = 'dataset_train_ff1000_mb.pt'
    dataset = get_dataset(dataset_name, device)
    train_iterator, valid_iterator = split_dataset(dataset)

    train_many_epochs(train_iterator=train_iterator,
                      valid_iterator=valid_iterator,
                      model=y2eq_trans_model,
                      device=device,
                      model_name='y2eq_transformer_mb.pt')
