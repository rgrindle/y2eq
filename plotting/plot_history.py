import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('../models/seq2seq_cnn_attention_model_history.csv').values
plt.plot(data[:, 0], label='train')
plt.plot(data[:, 1], label='val')
plt.xlabel('Epoch')
plt.ylabel('Symbolic loss')
plt.legend()
plt.savefig('plot_history_seq2seq_cnn_attention_model2.pdf')
