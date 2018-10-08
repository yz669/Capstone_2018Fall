# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 20:17:26 2018

@author: yizhu6
"""
import pandas as pd
import numpy as np
import time

# import W and reshape the data to get prepared, manually import
nsamples, nx, ny = W_new.shape
data_initial = W_new.reshape((nsamples,nx*ny))

# remove zero obs
df=pd.DataFrame(data_initial)
df['total']= df.sum(axis=1)
df_removezero = df[df.total != 0]
data =np.array(df_removezero.drop(['total'],axis=1))

#########################################

# determine K using elbow method

from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt


# euclidean
start_time = time.time()
sse1 = []
K = range(1,12)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(data)
    kmeanModel.fit(data)
    sse1.append(sum(np.min(cdist(data, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])

# Plot the elbow
plt.plot(K, sse1, 'bx-')
plt.xlabel('Number of K')
plt.ylabel('SSE')
plt.title('The Elbow Method Showing the Optimal K (Euclidean)')
plt.show()
print("--- %s seconds ---" % (time.time() - start_time))


# cosine
start_time = time.time()
sse = []
K = range(1,20)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(data)
    kmeanModel.fit(data)
    sse.append(sum(np.min(cdist(data, kmeanModel.cluster_centers_, 'cosine'), axis=1)) / X.shape[0])


# Plot the elbow
plt.plot(K, sse, 'bx-')
plt.xlabel('Number of K')
plt.ylabel('SSE')
plt.title('The Elbow Method Showing the Optimal K (Cosine)')
plt.show()



####################################




# do clustering
import nltk
from nltk.cluster.kmeans import KMeansClusterer
from sklearn.cluster import AgglomerativeClustering,KMeans

n_clusters = 5

#############################################

#KMeans - Euclidean
kclusterer = KMeansClusterer(n_clusters, distance=nltk.cluster.util.euclidean_distance)
clusters_table = kclusterer.cluster(data, assign_clusters=True)
pd.DataFrame(pd.Series(clusters_table).value_counts(), columns = ['NO. of clients']).T
#                  9   5   7   0   4   3   1   6   2   10  8 
#NO. of clients  3908  13   6   5   4   3   3   3   3   2   1


#KMeans - Cosine
kclusterer = KMeansClusterer(n_clusters, distance=nltk.cluster.util.cosine_distance)
clusters_table = kclusterer.cluster(data, assign_clusters=True)
pd.DataFrame(pd.Series(clusters_table).value_counts(), columns = ['NO. of clients']).T

centroid = kclusterer.means()

#                   3    0    4    2    1
#NO. of clients  1250  949  898  511  343


#                 5    8    1    9    3    2    0    10   7    4    6 
#NO. of clients  872  702  529  420  334  305  253  174  129  124  109

############################################





###########################################

#Hierarchical (Agglomerative) - Euclidean
model = AgglomerativeClustering(n_clusters, affinity='euclidean')
model.fit(data)
clusters_table = model.fit_predict(data)
pd.DataFrame(pd.Series(clusters_table).value_counts(), columns = ['NO. of clients']).T
#                  1   0   9   7   5   3   10  8   6   4   2 
#NO. of clients  3940   2   1   1   1   1   1   1   1   1   1



#Hierarchical (Agglomerative) - Cosine
model = AgglomerativeClustering(n_clusters, affinity='cosine', linkage='average')
model.fit(data)
clusters_table = model.fit_predict(data)
pd.DataFrame(pd.Series(clusters_table).value_counts(), columns = ['NO. of clients']).T
#                  1     4    5    2    10   8   0   3   6   9   7 
#NO. of clients  1484  1313  317  248  231  121  85  62  54  23  13


