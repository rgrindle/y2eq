"""
AUTHOR: Ryan Grindle

LAST MODIFIED: May 18, 2021

PURPOSE: Train y2eq so we can compare with MSR.

NOTES:

TODO:
"""
from srvgd.architecture.transformer.train_and_validate import get_dataset, train_many_epochs
from srvgd.architecture.transformer.y2eq_transformer import y2eq_trans_model

import torch
from torch.utils.data import DataLoader


if __name__ == '__main__':
    # Get number of trainable parameters
    num_params = sum(p.numel() for p in y2eq_trans_model.parameters() if p.requires_grad)
    print('Num trainable params:', num_params)

    # Get dataset
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataset = get_dataset('dataset_msr_train.pt', device)
    valid_dataset = get_dataset('dataset_msr_validation.pt', device)

    batch_size = 32
    train_iterator = DataLoader(train_dataset,
                                batch_size=batch_size,
                                shuffle=True)

    valid_iterator = DataLoader(valid_dataset,
                                batch_size=batch_size,
                                shuffle=False)

    # if load_model:
    #     load_checkpoint(torch.load("my_checkpoint.pth.tar"), y2eq_trans_model, optimizer)

    train_many_epochs(train_iterator=train_iterator,
                      valid_iterator=valid_iterator,
                      model=y2eq_trans_model,
                      device=device,
                      num_epochs=500,
                      model_name='y2eq_comp_msr.pt')
