import torch.nn as nn


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, include_coeff_values=False):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.include_coeff_values = include_coeff_values

    def forward(self, src, trg):

        # src = [batch size, src len]
        # trg = [batch size, trg len - 1] (<eos> token sliced off the end)

        # calculate z^u (encoder_conved) and (z^u + e) (encoder_combined)
        # encoder_conved is output from final encoder conv. block
        # encoder_combined is encoder_conved plus (elementwise) src embedding plus
        #  positional embeddings
        encoder_conved, encoder_combined = self.encoder(src)

        # encoder_conved = [batch size, src len, emb dim]
        # encoder_combined = [batch size, src len, emb dim]

        # calculate predictions of next words
        # output is a batch of predictions for each word in the trg sentence
        # attention a batch of attention scores across the src sentence for
        #  each word in the trg sentence
        if self.include_coeff_values:
            token_output, attention, coeff_output = self.decoder(trg, encoder_conved, encoder_combined)

            # output = [batch size, trg len - 1, output dim]
            # attention = [batch size, trg len - 1, src len]

            return token_output, attention, coeff_output
        else:
            output, attention = self.decoder(trg, encoder_conved, encoder_combined)

            # output = [batch size, trg len - 1, output dim]
            # attention = [batch size, trg len - 1, src len]

            return output, attention
