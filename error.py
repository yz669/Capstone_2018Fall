# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 22:56:48 2018

@author: Yi Zhu
"""


#calculate rmse
from math import sqrt
from sklearn.metrics import mean_squared_error

rms = sqrt(mean_squared_error(valid,forecast))
print(rms)





#calculate sse
from sklearn.cluster import KMeans
sse = []
K = range(1,20)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(data)
    kmeanModel.fit(data)
    sse.append(sum(np.min(cdist(data, kmeanModel.cluster_centers_, 'cosine'), axis=1)) / X.shape[0])




# confusion_matrix
import pandas as pd
from sklearn.metrics import confusion_matrix
# confusion matrix
actural = cluster1 # list
predict = cluster2 # list
# compute confusion matrix 
confusion_matrix = confusion_matrix(actural, predict)
# compute accuracy
accuracy = np.sum(confusion_matrix.diagonal())/float(np.sum(confusion_matrix))
accuracy





# ROC curve

import numpy as np
from sklearn import metrics
y = np.array(df_final.select('label').collect())
scores = np.array(df_final.select('prediction').collect())
fpr, tpr, thresholds = metrics.roc_curve(y, scores)
print(fpr)
print(tpr)


# compute roc_auc and then plot ROC curve
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

roc_auc = auc(fpr, tpr)

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='blue',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='darkorange', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
display(plt.show())


