import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch

# plot2eq_dict = torch.load('../models/checkpoint_plot2eq_dataset_train_ff1000_with_coeffs_resnet18_pretrainedFalse_epochs40.pt',
#                           map_location=torch.device('cpu'))
# print(plot2eq_dict.keys())
# print('epoch', plot2eq_dict['epoch'])
# print('epoch since improvement', plot2eq_dict['epochs_since_improvement'])
# print('val_loss', plot2eq_dict['val_loss'])
# exit()


y2eq_trans_dict = torch.load('../models/eq2y_transformer.pt',
                             map_location=torch.device('cpu'))
print(y2eq_trans_dict.keys())
data = np.vstack((y2eq_trans_dict['train_loss'],
                  y2eq_trans_dict['val_loss'])).T

print(data)
print('data.shape', data.shape)
# data = pd.read_csv('../models/train_history_dataset_train_ff1000_with_coeffs_v2_batchsize32_lr0.0001_clip1_layers10_includecoeffsTrue_includecoeffvaluesTrue_105.csv').values

plt.plot(data[:, 0], label='train')
plt.plot(data[:, 1], label='val')
plt.xlabel('Epoch')
# plt.ylabel('Numeric loss')
plt.ylabel('Symbolic loss')
plt.yscale('log')
# plt.ylim((0.17909955805904404, 0.38232180039518743))
plt.legend()
# plt.savefig('plot_history_seq2seq_cnn_attention_model5.pdf')
plt.show()
