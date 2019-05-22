
# coding: utf-8

# In[299]:


import pandas as pd

df_train_1 = pd.read_csv("/home/ehsana1998/Desktop/assignment_4_dataset/CIFAR-10-ByPart/CIFAR10_train_data.csv_split_aa.csv")
df_train_2 = pd.read_csv("/home/ehsana1998/Desktop/assignment_4_dataset/CIFAR-10-ByPart/CIFAR10_train_data.csv_split_ab.csv")
df_train_3 = pd.read_csv("/home/ehsana1998/Desktop/assignment_4_dataset/CIFAR-10-ByPart/CIFAR10_train_data.csv_split_ac.csv")
df_train_4 = pd.read_csv("/home/ehsana1998/Desktop/assignment_4_dataset/CIFAR-10-ByPart/CIFAR10_train_data.csv_split_ad.csv")
df_train_5 = pd.read_csv("/home/ehsana1998/Desktop/assignment_4_dataset/CIFAR-10-ByPart/CIFAR10_train_data.csv_split_ae.csv")
df_train = pd.concat([df_train_1, df_train_2, df_train_3, df_train_4, df_train_5]).values

y_train_1 = pd.read_csv("/home/ehsana1998/Desktop/assignment_4_dataset/CIFAR-10-ByPart/CIFAR10_train_label.csv_split_aa.csv")
y_train_2 = pd.read_csv("/home/ehsana1998/Desktop/assignment_4_dataset/CIFAR-10-ByPart/CIFAR10_train_label.csv_split_ab.csv")
y_train_3 = pd.read_csv("/home/ehsana1998/Desktop/assignment_4_dataset/CIFAR-10-ByPart/CIFAR10_train_label.csv_split_ac.csv")
y_train_4 = pd.read_csv("/home/ehsana1998/Desktop/assignment_4_dataset/CIFAR-10-ByPart/CIFAR10_train_label.csv_split_ad.csv")
y_train_5 = pd.read_csv("/home/ehsana1998/Desktop/assignment_4_dataset/CIFAR-10-ByPart/CIFAR10_train_label.csv_split_ae.csv")
y_train = (pd.concat([y_train_1, y_train_2, y_train_3, y_train_4, y_train_5])).values
y_train = [x[0] for x in y_train]

testData = pd.read_csv("/home/ehsana1998/Desktop/assignment_4_dataset/CIFAR-10-ByPart/CIFAR10_val_data.csv").values
y_test = pd.read_csv("/home/ehsana1998/Desktop/assignment_4_dataset/CIFAR-10-ByPart/CIFAR10_val_label.csv").values
y_test = [y[0] for y in y_test]


# In[156]:


df_train.info()


# In[220]:


df_train.shape


# In[2]:


import numpy as np
from matplotlib import pyplot as plt

M = [
  [(0, 0, 255), (0, 255, 0)],
  [(255, 0, 0), (0, 0, 0)]
]

image = np.array(M, dtype=np.uint8)[...,::-1]
image_transp = np.transpose(image, (1, 0, 2))

# plt.imshow(image_transp, interpolation='none')
image_transp.shape


# In[221]:


pic = df_train[10]

def data2image(pic):
    
    image = []
    for i in range(32):
        currRow = []
        for t in range(32):
            currPixel = []
            for x in range(3):
                currPixel.append(pic[96 * i + 3 * t + x])
            currRow.append(currPixel)
        image.append(currRow)
    
    return image

image = data2image(pic)
image = np.array(image, dtype=np.uint8)[...,::-1]
image_transp = np.transpose(image, (1, 0, 2))

plt.imshow(image, interpolation='none')


# In[300]:


from sklearn.decomposition import PCA

pca = PCA(n_components=0.75, whiten=True).fit(df_train)


# In[301]:


X_train_pca = pca.transform(df_train)


# In[225]:


X_train_pca.shape


# In[302]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=6)


# In[314]:


X_train = df_train
X_train.shape


# In[162]:


knn.fit(X_train, y_train)


# In[163]:


y_pred = knn.predict(testData)


# In[164]:


y_test = np.asarray(y_test)


# In[165]:


correctNo = 0
for index, prediction in enumerate(y_pred):
    if prediction == y_test[index]:
        correctNo += 1
correctNo


# In[303]:


testData_pca = pca.transform(testData)


# In[304]:


knn.fit(X_train_pca, y_train)
y_pred_pca = knn.predict(testData_pca)


# In[305]:


correctNo = 0
for index, prediction in enumerate(y_pred_pca):
    if prediction == y_test[index]:
        correctNo += 1
correctNo


# In[264]:


from skimage.color import rgb2gray
image = data2image(df_train[10])
new_image = rgb2grey(np.asarray(image))
new_image[0][0]
# plt.imshow(new_image, interpolation='none')


# In[250]:


from skimage.color import rgb2grey
new_image = rgb2grey(np.asarray(image))
(np.asarray(new_image)).shape
# plt.imshow(new_image, interpolation='none')


# In[270]:


def image2data(pic):
    data = []
    for row in pic:
        for pixel in row:
            data.append(pixel)
                
    return data


# In[241]:


data = image2data(new_image)
np.asarray(data).shape


# In[288]:


def turnImages2grayscale(data):
    
    rslData = []
    for imageData in data:
        image = data2image(imageData)
        new_image = rgb2grey(np.asarray(image))
        new_imageData = np.asarray(image2data(new_image))
#         print(new_imageData.shape)
        rslData.append(new_imageData)
    
    return rslData


# In[289]:


X_train_n = turnImages2grayscale(X_train)


# In[292]:


knn.fit(X_train_n, y_train)


# In[294]:


y_pred = knn.predict(turnImages2grayscale(testData))
y_test = np.asarray(y_test)
correctNo = 0
for index, prediction in enumerate(y_pred):
    if prediction == y_test[index]:
        correctNo += 1
correctNo


# In[298]:


pca = PCA(n_components=0.75, whiten=True).fit(X_train_n)
X_train_pca_n = pca.transform(X_train_n)
testData_pca_n = pca.transform(turnImages2grayscale(testData))
knn.fit(X_train_pca_n, y_train)
y_pred_pca = knn.predict(testData_pca_n)
correctNo = 0
for index, prediction in enumerate(y_pred_pca):
    if prediction == y_test[index]:
        correctNo += 1
correctNo


# In[308]:


from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection
from sklearn.decomposition import PCA, FastICA
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF
from sklearn.cluster import FeatureAgglomeration

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Ridge

from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
import pickle

import warnings
warnings.filterwarnings('ignore')

import random
random.seed(1729)


# In[315]:


n_comp = 12

# tSVD
tsvd = TruncatedSVD(n_components=n_comp, random_state=420)
tsvd_results_train = tsvd.fit_transform(X_train)
tsvd_results_test = tsvd.transform(testData)

# PCA
pca = PCA(n_components=n_comp, random_state=420)
pca2_results_train = pca.fit_transform(X_train)
pca2_results_test = pca.transform(testData)

# ICA
ica = FastICA(n_components=n_comp, random_state=420)
ica2_results_train = ica.fit_transform(X_train)
ica2_results_test = ica.transform(testData)

# GRP
grp = GaussianRandomProjection(n_components=n_comp, eps=0.1, random_state=420)
grp_results_train = grp.fit_transform(X_train)
grp_results_test = grp.transform(testData)

# SRP
srp = SparseRandomProjection(n_components=n_comp, dense_output=True, random_state=420)
srp_results_train = srp.fit_transform(X_train)
srp_results_test = srp.transform(testData)

# NMF
nmf = NMF(n_components=n_comp, init='nndsvdar', random_state=420)
nmf_results_train = nmf.fit_transform(X_train)
nmf_results_test = nmf.transform(testData)

# FAG
fag = FeatureAgglomeration(n_clusters=n_comp, linkage='ward')
fag_results_train = fag.fit_transform(X_train)
fag_results_test = fag.transform(testData)


# In[316]:


knn.fit(tsvd_results_train, y_train)
y_pred_pca = knn.predict(tsvd_results_test)
correctNo = 0
for index, prediction in enumerate(y_pred_pca):
    if prediction == y_test[index]:
        correctNo += 1
correctNo


# In[317]:


knn.fit(pca2_results_train, y_train)
y_pred_pca = knn.predict(pca2_results_test)
correctNo = 0
for index, prediction in enumerate(y_pred_pca):
    if prediction == y_test[index]:
        correctNo += 1
correctNo


# In[318]:


knn.fit(ica2_results_train, y_train)
y_pred_pca = knn.predict(pca2_results_test)
correctNo = 0
for index, prediction in enumerate(ica2_results_test):
    if prediction == y_test[index]:
        correctNo += 1
correctNo


# In[319]:


knn.fit(grp_results_train, y_train)
y_pred_pca = knn.predict(grp_results_test)
correctNo = 0
for index, prediction in enumerate(y_pred_pca):
    if prediction == y_test[index]:
        correctNo += 1
correctNo


# In[320]:


knn.fit(srp_results_train, y_train)
y_pred_pca = knn.predict(srp_results_test)
correctNo = 0
for index, prediction in enumerate(y_pred_pca):
    if prediction == y_test[index]:
        correctNo += 1
correctNo


# In[321]:


knn.fit(nmf_results_train, y_train)
y_pred_pca = knn.predict(nmf_results_test)
correctNo = 0
for index, prediction in enumerate(y_pred_pca):
    if prediction == y_test[index]:
        correctNo += 1
correctNo


# In[322]:


knn.fit(fag_results_train, y_train)
y_pred_pca = knn.predict(fag_results_test)
correctNo = 0
for index, prediction in enumerate(y_pred_pca):
    if prediction == y_test[index]:
        correctNo += 1
correctNo


# In[338]:


# tSVD

max_c = 0
rsl_n_comp = 12
for n_comp in range(20, 40):
    tsvd = TruncatedSVD(n_components=n_comp, random_state=420)
    tsvd_results_train = tsvd.fit_transform(X_train)
    tsvd_results_test = tsvd.transform(testData)
    knn.fit(tsvd_results_train, y_train)
    y_pred_pca = knn.predict(tsvd_results_test)
    correctNo = sum(1 for index, prediction in enumerate(y_pred_pca) if prediction == y_test[index])
    if correctNo > max_c:
        max_c = correctNo
        rsl_n_comp = n_comp


# In[339]:


rsl_n_comp


# In[340]:


max_c


# In[349]:


max_c = 0
rsl_n_comp = 12
for n_comp in range(18, 40):
    pca = PCA(n_components=n_comp, random_state=420)
    pca2_results_train = pca.fit_transform(X_train)
    pca2_results_test = pca.transform(testData)
    knn.fit(pca2_results_train, y_train)
    y_pred_pca = knn.predict(pca2_results_test)
    correctNo = sum(1 for index, prediction in enumerate(y_pred_pca) if prediction == y_test[index])
    if correctNo > max_c:
        max_c = correctNo
        rsl_n_comp = n_comp


# In[350]:


rsl_n_comp


# In[351]:


max_c


# In[361]:



max_c = 0
rsl_n_comp = 12
for n_comp in range(34, 40):
    nmf = NMF(n_components=n_comp, init='nndsvdar', random_state=420)
    nmf_results_train = nmf.fit_transform(X_train)
    nmf_results_test = nmf.transform(testData)
    knn.fit(nmf_results_train, y_train)
    y_pred_pca = knn.predict(nmf_results_test)
    correctNo = sum(1 for index, prediction in enumerate(y_pred_pca) if prediction == y_test[index])
    if correctNo > max_c:
        max_c = correctNo
        rsl_n_comp = n_comp


# In[362]:


rsl_n_comp


# In[363]:


max_c

