
# coding: utf-8

# # PCA and Random Projection
# 
# هر دوی این ها روش هایی برای کاهش ابعاد مجموعه داده هستند که توسط روابط ریاضی ابعاد مجموعه داده را کاهش می دهند. دلیل اینکه می توانند خطا را کم کنند این است که کم شدن ابعاد به کم شدن پیچیدگی مدل می انجامد و عملا قسمت هایی که از مجموعه داده تعیین کننده نیست را کنار می گذارد و نکته مثبت دیگر این روش ها این است که باعث کم شدن مقدار داده می شوند وسرعت عملکرد مدل بالا می رود

# # Turn the images to grayscale
# 
# این الگوریتم سه عدد قرمز- سبز و آبی یک پیکسل را می گیرد و یک عدد به آن پیکسل نسبت می دهد. این الگوریتم می تواند در بهبود مدل کمک کند به این دلیل که رنگ های بی معنی داخل تصویر بی اثر می شوند و فقط خود شکل اهمیت پیدا می کند

# # Augmentation
# 
# این الگوریتم زمانی به درد می خورد که حس کنیم داده ی ما کم است و بخواهیم داده تولید کنیم. می توان با تغییر کوجک بر روی عکس ها تصاویر جدید با همان برچسب های قبلی ایجاد کرد

# # Import Data

# In[1]:


import pandas as pd

df_train = pd.read_csv("CIFAR10_train_data.csv").values
y_train = [y[0] for y in pd.read_csv("CIFAR10_train_label.csv").values]

testData = pd.read_csv("CIFAR10_val_data.csv").values
y_test = [y[0] for y in pd.read_csv("CIFAR10_val_label.csv").values]


# # PCA with knn and random forest

# In[13]:



# PCA

from sklearn.decomposition import PCA

n_comp = 23
pca = PCA(n_components=n_comp, random_state=390)
pca_results_train = pca.fit_transform(df_train)
pca_results_test = pca.transform(testData)

#knn

from sklearn.neighbors import KNeighborsClassifier

k = 6
model = KNeighborsClassifier(n_neighbors=k)
model.fit(pca_results_train, y_train)
score = model.score(pca_results_test, y_test)
print("PCA with KNN accuracy = %.2f%%" % (score * 100))

#PCA

from sklearn.ensemble import RandomForestClassifier

treesNo = 100
maxAccuracy = 0
for tn in range(100, 146):
    model = RandomForestClassifier(n_estimators=tn)
    model.fit(pca_results_train, y_train)
    score = model.score(pca_results_test, y_test)
    if score > maxAccuracy:
        maxAccuracy = score
        treesNo = tn
        
print("max accuracy of PCA with Random Forest = %.2f%% , trees number =%d " % ((maxAccuracy * 100), treesNo))        


# # Random Projection

# In[6]:


from sklearn.ensemble import RandomForestClassifier
n_comp = 12

# GaussianRandomProjection

from sklearn.random_projection import GaussianRandomProjection

grp = GaussianRandomProjection(n_components=n_comp, eps=0.1, random_state=420)
grp_results_train = grp.fit_transform(df_train)
grp_results_test = grp.transform(testData)

treesNo = 141
model = RandomForestClassifier(n_estimators=treesNo)
model.fit(grp_results_train, y_train)
score = model.score(grp_results_test, y_test)
print("GaussianRandomProjection with random forest accuracy = %.2f%%" % (score * 100))

# SparseRandomProjection

from sklearn.random_projection import SparseRandomProjection

srp = SparseRandomProjection(n_components=n_comp, dense_output=True, random_state=420)
srp_results_train = srp.fit_transform(df_train)
srp_results_test = srp.transform(testData)

model.fit(srp_results_train, y_train)
score = model.score(srp_results_test, y_test)
print("SparseRandomProjection with random forest accuracy = %.2f%%" % (score * 100))


# # Turn the images to grayscale 

# In[3]:



import numpy as np

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

def image2data(pic):
    data = []
    for row in pic:
        for pixel in row:
            data.append(pixel)

    return data

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def turnImages2grayscale(data):
    
    rslData = []
    for imageData in data:
        image = data2image(imageData)
        new_image = rgb2gray(np.asarray(image))
        new_imageData = np.asarray(image2data(new_image))
        rslData.append(new_imageData)
    
    return rslData

# df_train_new = turnImages2grayscale(df_train)
# testData_new = turnImages2grayscale(testData)


# In[11]:



treesNo = 141
model = RandomForestClassifier(n_estimators=treesNo)
model.fit(df_train_new, y_train)
score = model.score(testData_new, y_test)
print("Grayscale images with Random Forest accuracy = %.2f%%" % (score * 100))

from sklearn.decomposition import PCA

n_comp = 23
pca = PCA(n_components=n_comp, random_state=390)
pca_results_train_new = pca.fit_transform(df_train_new)
pca_results_test_new = pca.transform(testData_new)

model.fit(pca_results_train_new, y_train)
score = model.score(pca_results_test_new, y_test)
print("Grayscale images and PCA with Random Forest accuracy = %.2f%%" % (score * 100))


# # Augmentation and PCA

# In[8]:


from copy import deepcopy

def add_noise(img):
    noise = np.random.randint(5, size = (164, 278, 4), dtype = 'uint8')
    HEIGHT = np.asarray(img).shape[0]
    WIDTH = np.asarray(img).shape[1]
    DEPTH = np.asarray(img).shape[2]
    for i in range(WIDTH):
        for j in range(HEIGHT):
            for k in range(DEPTH):
                if (img[i][j][k] != 255):
                    img[i][j][k] += noise[i][j][k]
    
    return img
                
def shift_left(img):
    HEIGHT = np.asarray(img).shape[0]
    WIDTH = np.asarray(img).shape[1]
    for i in range(HEIGHT, 1, -1):
        for j in range(WIDTH):
            if (i < HEIGHT-20):
                img[j][i] = img[j][i-20]
            elif (i < HEIGHT-1):
                img[j][i] = 0
    return img

def augmentate(images):
    
    flipped_images = np.asarray([np.fliplr(deepcopy(img)) for img in images])
    shifted_l_images = np.asarray([shift_left(deepcopy(img)) for img in images])
    added_noise_images = np.asarray([add_noise(deepcopy(img)) for img in images])
    print(images.shape, flipped_images.shape, shifted_l_images.shape, added_noise_images.shape)
    rsl_images = np.concatenate([images, flipped_images, shifted_l_images, added_noise_images])
    return rsl_images

sample_df_train = df_train[:10000]
sample_y_train = y_train[:10000]

images = [data2image(data) for data in sample_df_train]
augmentated_images = augmentate(np.asarray(images))
augmentated_images_labels = np.concatenate([sample_y_train, sample_y_train, sample_y_train, sample_y_train])


# In[12]:


def image2data_v1(pic):
    data = []
    for row in pic:
        for r,g,b in row:
            data.append(r)
            data.append(g)
            data.append(b)

    return data

from sklearn.decomposition import PCA

n_comp = 23
pca = PCA(n_components=n_comp, random_state=390)
pca_results_train = pca.fit_transform([image2data_v1(image) for image in augmentated_images])
pca_results_test = pca.transform(testData)

from sklearn.ensemble import RandomForestClassifier

treesNo = 141
model = RandomForestClassifier(n_estimators=treesNo)
model.fit(pca_results_train, augmentated_images_labels)
score = model.score(pca_results_test, y_test)
print("Augmentation and PCA with Random Forest accuracy = %.2f%%" % (score * 100))


# # TruncatedSVD

# In[14]:



model = RandomForestClassifier(n_estimators=treesNo)

from sklearn.decomposition import TruncatedSVD

maxAccuracy = 0
rsl_n_comp = 15
for n_comp in range(15, 40):
    tsvd = TruncatedSVD(n_components=n_comp, random_state=400)
    tsvd_results_train = tsvd.fit_transform(df_train)
    tsvd_results_test = tsvd.transform(testData)
    model.fit(tsvd_results_train, y_train)
    score = model.score(tsvd_results_test, y_test)
    if score > maxAccuracy:
        maxAccuracy = score
        rsl_n_comp = n_comp
        
print("max accuracy of PCA with Random Forest = %.2f%% , components number =%d " % ((maxAccuracy * 100), rsl_n_comp))


# In[2]:


from sklearn.ensemble import RandomForestClassifier

treesNo = 141
finalModel = RandomForestClassifier(n_estimators=treesNo)

from sklearn.decomposition import TruncatedSVD

tsvd = TruncatedSVD(n_components=27, random_state=400)
tsvd_results_train = tsvd.fit_transform(df_train)
tsvd_results_test = tsvd.transform(testData)
finalModel.fit(tsvd_results_train, y_train)
score = finalModel.score(tsvd_results_test, y_test)

data4pred = pd.read_csv("CIFAR10_test_data.csv")
tsvd_results_data4pred = tsvd.transform(data4pred)
finalPred = finalModel.predict(tsvd_results_data4pred)


# In[12]:


print("max accuracy = %.2f%%" % (score * 100))


# In[9]:


len(data4pred)


# In[7]:


output = {'id': [_id for _id in range(1, len(finalPred) + 1)],
         'predict': [predict for predict in finalPred]}
df_output = pd.DataFrame(output)
df_output.to_csv("810197668.csv", index=False)

