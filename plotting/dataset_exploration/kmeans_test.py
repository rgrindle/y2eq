import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

import itertools


semantics = np.array([[1, 2], [1, 4], [1, 0],
                      [10, 2], [10, 4], [10, 0]])
kmeans = KMeans(n_clusters=2, random_state=0).fit(semantics)
print(kmeans.labels_)

test_semantics = [[0, 0], [12, 3]]
# pred = kmeans.predict(test_semantics)
# print(pred)
# print(kmeans.cluster_centers_)

# draw boundaries
xmin = -1
xmax = 13
ymin = -1
ymax = 13
x = np.linspace(xmin, xmax, 10)
y = np.linspace(ymin, ymax, 10)
mesh = np.array(list(itertools.product(x, y)))
print(mesh)
print(mesh.shape, semantics.shape)
pred = kmeans.predict(mesh)
print(pred.shape)
pred = pred.reshape((len(x), len(y))).T
print(pred.shape)
plt.imshow(pred, interpolation='nearest',
           extent=(xmin, xmax, ymin, ymax),
           cmap=plt.cm.tab10, vmin=0, vmax=10,
           origin='lower', alpha=0.5)
# for x, p in zip(mesh, pred):
#     print(x, p)
#     p = kmeans.predict([x])[0]
#     plt.plot(x[0], x[1], 'd', color='C{}'.format(p))


for x in semantics:
    plt.plot(x[0], x[1], '.', color='C{}'.format(kmeans.predict([x])[0]))
for x in test_semantics:
    plt.plot(x[0], x[1], 'x', color='C{}'.format(kmeans.predict([x])[0]))

# plt.legend()
plt.show()
