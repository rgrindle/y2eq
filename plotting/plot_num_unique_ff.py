import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

models = ['y2eq-fixed-fixed', 'plot2eq-fixed-fixed',
          'y2eq-transformer-fixed-fixed', 'plot2eq-transformer-fixed-fixed',
          'y2eq-transformer-no-coeffs-fixed-fixed']

num_unique_ff = []
for m in models:
    data = pd.read_csv('../eval_'+m+'/01_predicted_ff.csv', header=None).values
    unique_data = np.unique(data)
    num_unique_ff.append(unique_data.shape[0])

plt.bar(range(len(models)), num_unique_ff)
for i, v in enumerate(num_unique_ff):
    plt.text(i, v/2, str(v), color='white', fontweight='bold', ha='center', va='center')

plt.xticks(range(len(models)), models, rotation=45)
plt.ylabel('Number of unique\nfunctional forms output')
plt.tight_layout()
plt.show()
