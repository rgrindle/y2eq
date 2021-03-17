"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Mar 16, 2021

PURPOSE: Evaluate y2eq WITH teacher forcing.

NOTES:

TODO:
"""
import torch


def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch[0]
            trg = batch[1]

            # Get only y (remove x0, x1)
            # This data is already order by x0 then by x1,
            # so no need to order it here.
            # Then, make sure that shape is correct
            # (batch_size, num_y_values, 1)
            src = src[:, 2:, :]
            src = src.permute(0, 2, 1)

            output, _ = model(src, trg[:, :-1])

            # trg = [trg len, batch size]
            # output = [trg len, batch size, output dim]

            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)

            # trg = [(trg len - 1) * batch size]
            # output = [(trg len - 1) * batch size, output dim]

            loss = criterion(output, trg)
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)
