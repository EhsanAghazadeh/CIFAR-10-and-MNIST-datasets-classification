
# coding: utf-8

# In[5]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets


# In[2]:



X = iris.data[:, :2]
y = iris.target

h = 0.2
# Create color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

for weights in ['uniform', 'distance']:
    # we create an instance of Neighbours Classifier and fit the data.
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
    clf.fit(X, y)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
                edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("3-Class classification (k = %i, weights = '%s')"
              % (n_neighbors, weights))

plt.show()


# In[1]:


import warnings
warnings.filterwarnings('ignore')

import pandas as pd

X_train = pd.read_csv("/home/ehsana1998/Desktop/MNIST-dataset/MNIST/train_data.csv").values
y_train = pd.read_csv("/home/ehsana1998/Desktop/MNIST-dataset/MNIST/train_label.csv").values


# In[2]:


X_test = pd.read_csv("/home/ehsana1998/Desktop/assignment_4_dataset/dataset/MNIST/test_data.csv").values
y_test = pd.read_csv("/home/ehsana1998/Desktop/assignment_4_dataset/dataset/MNIST/test_label.csv").values


# In[15]:


X_train[10].reshape(28,28).shape


# In[17]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.imshow(X_train[400].reshape((28,28)), cmap="Greys")


# # K-Nearest Neighbors
# 
# نحوه ی عملکرد این الگوریتم به این صورت است که وقتی می خواهد پیش بینی کند داده ی جدید متعلق به چه کلاسی است نقطه ی داده ی جدید را در میان داده های در زمان یادگیری قرار می دهد و به تعدادی مشخص داده های با کمترین فاصله نسبت به این نقطه را پیدا می کند و کلاس این داده آن کلاسی تشخیص می دهد که بیشترین تعداد نقاط را در بین این نقاط با کمترین فاصله داشته باشد

# In[5]:


from __future__ import print_function
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import numpy as np


# In[25]:



kVals = range(1, 30, 2)
accuracies = []
trainAccuracies = []

for k in range(1, 30, 2):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    
    score = model.score(X_test, y_test)
    score_train = model.score(X_train, y_train)
    print("k=%d, test data accuracy=%.2f%%, train data accuracy=%.2f%%" % (k, score * 100, score_train * 100))
    accuracies.append(score)
    trainAccuracies.append(score_train)
    
i = np.argmax(accuracies)
print("k=%d achieved highest accuracy of %.2f%% on validation data" % (kVals[i],accuracies[i] * 100))


# In[26]:


accuracies = [(accuracy * 100) for accuracy in accuracies]
trainAccuracies = [(accuracy * 100) for accuracy in trainAccuracies]


# In[27]:


fig, axs = plt.subplots()
axs.plot(kVals, accuracies, color='b', label='Test accuracy')
axs.plot(kVals, trainAccuracies, color='r', label='Train accuracy')
axs.set_title("K Nearest Neighbors accuracy per neighbors number")
axs.set_xlabel("Neighbors number")
axs.set_ylabel("Accuracy")
plt.show()


# با توجه به نمودار با افزایش تعداد همسایه ها دقت داده آموزش کم می شود و وقتی تعداد همسایه ها کم است خطر وقوع اورفیت وجود دارد و تعداد همسایه ها باید از روی خروجی داده تست تعیین گردد

# یکی از مشکلات این الگوریتم این است که به حافظه ی زیادی نیاز دارد و باید همه ی داده ی زمان یادگیری را نگه داریم

# # Decision Tree
# 
# این الگوریتم به این شکل است که در زمان یادگیری تعدادی سوال طرح می شود که پاسخ این سوال ها ما را به پیش بینی صحیح می رساند. حال اینکه این سوال ها چه سوال هایی باشند از روی داده های ترین و با توجه به نظریه اطلاعات مشخص می شود 

# In[28]:


from sklearn.tree import DecisionTreeClassifier

dVals = range(1, 30, 1)
accuracies = []
trainAccuracies = []

for d in range(1, 30, 1):
    model = DecisionTreeClassifier(max_depth=d)
    model.fit(X_train, y_train)
    
    score = model.score(X_test, y_test)
    score_train = model.score(X_train, y_train)
    print("depth=%d, test data accuracy=%.2f%%, train data accuracy=%.2f%%" % (d, score * 100, score_train * 100))
    trainAccuracies.append(score_train)
    accuracies.append(score)
    
i = np.argmax(accuracies)
print("depth=%d achieved highest accuracy of %.2f%% on validation data" % (dVals[i],accuracies[i] * 100))


# In[29]:


accuracies = [(accuracy * 100) for accuracy in accuracies]
trainAccuracies = [(accuracy * 100) for accuracy in trainAccuracies]


# In[30]:


fig, axs = plt.subplots()
axs.plot(dVals, accuracies, color='b', label='Test accuracy')
axs.plot(dVals, trainAccuracies, color='r', label='Train accuracy')
axs.set_title("Decision Tree accuracy per max depth")
axs.set_xlabel("max depth")
axs.set_ylabel("Accuracy")
plt.show()


# In[31]:


import graphviz

model = tree.DecisionTreeClassifier(max_depth= 12)
model.fit(X_train, y_train)
dot_data = tree.export_graphviz(model, out_file=None)
graph = graphviz.Source(dot_data)
graph.render("mnist")


# با توجه به داده ها با افزایش بیشنه ارتفاع درخت رفته رفته دچار اورفیت می شویم

# In[62]:


from sklearn import tree
from IPython.display import Image
# import pydotplus

model = DecisionTreeClassifier(max_depth=15)
model.fit(X_train, y_train)
# model.
# dot_data = StringIO()
tree.export_graphviz(model, out_file="tree.dot")
# graph = pydotplus.graph_from_dot_data("tree.dot".getvalue())
get_ipython().system('dot -Tpng tree.dot -o tree.png -Gdpi=600')


# # Random Forest
# 
# در این الگوریتم همان طور که از اسمش پیداست جنگلی می سازیم. به واقع این الگوریتم تعدادی درخت تصمیم می سازد که موازی با هم کار می کنند و نتایج آن ها را با هم ادغام می کنیم تا پیش بینی درست تر و پایدارتری داشته باشیم 

# In[32]:


from sklearn.ensemble import RandomForestClassifier

dVals = range(1, 30, 1)
accuracies = []
trainAccuracies = []

for d in range(1, 30, 1):
    model = RandomForestClassifier(max_depth=d)
    model.fit(X_train, y_train)
    
    score = model.score(X_test, y_test)
    train_score = model.score(X_train, y_train)
    print("depth=%d, test accuracy=%.2f%%, train accuracy =%.2f%%" % (d, score * 100, train_score * 100))
    accuracies.append(score)
    trainAccuracies.append(train_score)
    
i = np.argmax(accuracies)
print("depth=%d achieved highest accuracy of %.2f%% on validation data" % (dVals[i],accuracies[i] * 100))


# In[34]:


accuracies = [(accuracy * 100) for accuracy in accuracies]
trainAccuracies = [(accuracy * 100) for accuracy in trainAccuracies]


# In[35]:


fig, axs = plt.subplots()
axs.plot(dVals, accuracies, color='b', label='Test accuracy')
axs.plot(dVals, trainAccuracies, color='r', label='Train accuracy')
axs.set_title("Random Forest accuracy per max depth")
axs.set_xlabel("max depth")
axs.set_ylabel("Accuracy")
plt.show()


# In[36]:


from sklearn.ensemble import RandomForestClassifier

nVals = range(1, 150, 5)
accuracies = []
trainAccuracies = []

for n in range(1, 150, 5):
    model = RandomForestClassifier(max_depth=11, n_estimators=n)
    model.fit(X_train, y_train)
    
    score = model.score(X_test, y_test)
    train_score = model.score(X_train, y_train)
    print("number of trees=%d, test accuracy=%.2f%%, train accuracy=%.2f%%" % (n, score * 100, train_score * 100))
    accuracies.append(score)
    trainAccuracies.append(train_score)
    
i = np.argmax(accuracies)
print("number of trees=%d achieved highest accuracy of %.2f%% on validation data" % (nVals[i],accuracies[i] * 100))


# In[37]:


accuracies = [(accuracy * 100) for accuracy in accuracies]
trainAccuracies = [(accuracy * 100) for accuracy in trainAccuracies]


# In[38]:


fig, axs = plt.subplots()
axs.plot(nVals, accuracies, color='b', label='Test accuracy')
axs.plot(nVals, trainAccuracies, color='r', label='Train accuracy')
axs.set_title("Random Forest accuracy per number of trees")
axs.set_xlabel("number of trees")
axs.set_ylabel("Accuracy")
plt.show()


# In[6]:


from sklearn.ensemble import RandomForestClassifier

sVals = range(2, 30, 1)
accuracies = []
trainAccuracies = []

for s in range(2, 30, 1):
    model = RandomForestClassifier(max_depth=11, n_estimators=136, min_samples_split=s)
    model.fit(X_train, y_train)
    
    score = model.score(X_test, y_test)
    train_score = model.score(X_train, y_train)
    print("min samples split=%d, test accuracy=%.2f%%, train accuracy=%.2f%%" % (s, score * 100, train_score * 100))
    accuracies.append(score)
    trainAccuracies.append(train_score)
    
i = np.argmax(accuracies)
print("min samples split=%d achieved highest accuracy of %.2f%% on validation data" % (sVals[i],accuracies[i] * 100))


# In[7]:


accuracies = [(accuracy * 100) for accuracy in accuracies]
trainAccuracies = [(accuracy * 100) for accuracy in trainAccuracies]


# In[9]:


fig, axs = plt.subplots()
axs.plot(sVals, accuracies, color='b', label='Test accuracy')
axs.plot(sVals, trainAccuracies, color='r', label='Train accuracy')
axs.set_title("Random Forest accuracy per min samples split")
axs.set_xlabel("min samples split")
axs.set_ylabel("Accuracy")
plt.show()


# In[10]:


from sklearn.ensemble import RandomForestClassifier

fVals = range(1, 30, 1)
accuracies = []
trainAccuracies = []

for f in range(1, 30, 1):
    model = RandomForestClassifier(max_depth=11, n_estimators=136, min_samples_split=4, max_features=f)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    train_score = model.score(X_train, y_train)
    print("max features=%d, test accuracy=%.2f%%, train accuracy=%.2f%%" % (f, score * 100, train_score * 100))
    accuracies.append(score)
    trainAccuracies.append(train_score)
    
i = np.argmax(accuracies)
print("max features=%d achieved highest accuracy of %.2f%% on validation data" % (fVals[i],accuracies[i] * 100))


# In[11]:


accuracies = [(accuracy * 100) for accuracy in accuracies]
trainAccuracies = [(accuracy * 100) for accuracy in trainAccuracies]


# In[12]:


fig, axs = plt.subplots()
axs.plot(fVals, accuracies, color='b', label='Test accuracy')
axs.plot(fVals, trainAccuracies, color='r', label='Train accuracy')
axs.set_title("Random Forest accuracy per max features")
axs.set_xlabel("max features")
axs.set_ylabel("Accuracy")
plt.show()


# # N-estimators
# 
# این همان تعداد درخت ها در جنگل است که به صورت موازی با هم کار می کنند و نتیجه ادغام نتایج این تعداد درخت خواهد شد

# # Logistic Regression
# 
# این الگوریتم همان رگرسیون است با این تفاوت که پشت خروجی رگرسیون یک تابع لجستیک مانند سیگموید یا ساین می آید که باعث می شود خروجی نهایی در یک بازه ی محدود باشد و هر عددی نمی تواند باشد. این الگوریتم سه دسته دارد که عبارتند از رگرسیون لجستیک باینری مثل تشخیص امن بودن یا نبودن ایمیل-رگرسیون لجستیک چندتایی که خروجی یک کلاس از بین چند کلاس است و رگرسیون لجستیک ترتیبی که خروجی یک کلاس از بین چند کلاس با ترتیب است

# In[92]:


from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)

score = model.score(X_test, y_test)
print("test data accuracy=%.2f%%" % (score * 100))

score = model.score(X_train, y_train)
print("train data accuracy=%.2f%%" % (score * 100))


# # K-means
# 
# این الگوریتم سه مرحله دارد: ۱-تعداد مشخصی نقطه را به صورت رندوم از بین داده ها انتخاب می کند به عنوان میانگین های اولیه ۲- هر نقطه داده را به میانگینی با کمترین فاصله اقلیدسی ارجاع می دهد.حال میانگین این دسته ها به دست می آورد و به عنوان میانگین های جدید قرار می دهد ۳- فرایند مراحل یک و دو را به تعدادی معلوم تکرار می کنیم

# In[27]:


from collections import Counter

def get_max_freq(lst):
    x = Counter(lst) 
    return x.most_common(1)[0][0]

from sklearn.cluster import KMeans

model = KMeans(n_clusters=10)
model.fit(X_train)
kmeans_labels = model.labels_

primitiveClasses = [[] for i in range(10)]
for i in range(X_train.shape[0]):
    primitiveClasses[kmeans_labels[i]].append(int(y_train[i]))
    
    
for i in range(10):
    primitiveClasses[i] = get_max_freq(primitiveClasses[i])
    
primitiveClasses


# In[28]:


fig = plt.figure()
for index, val in enumerate(model.cluster_centers_.reshape(10, 28, 28)):
    fig.add_subplot(2, 5, index + 1)
    img = val
    plt.title(primitiveClasses[index])
    plt.imshow(img, cmap="Greys")


# In[35]:


pred = model.predict(X_test)
for i in range(len(pred)):
    pred[i] = primitiveClasses[pred[i]]
    
correctedNo = sum(1 for index, val in enumerate(pred) if val == int(y_test[index]))
print("accuracy=%.2f%%" % (correctedNo / len(pred) * 100))


# با افزایش تعداد میانگین ها دقت بالا می رود و رفته رفته دقتش به دقت نزدیکترین همسایه نزدیک می شود و وقتی تعداد میانگین ها برابر تعداد داده ها شود همان نزدیک ترین همسایه خواهد بود. زمان این اگوریتم بسته به تعداد میانگین هاست اما در نزدیک ترین همسایه زمان بسته به تعداد داده های آموزش است

# # بهترین الگوریتم: جنگل تصادفی

# In[110]:


model = RandomForestClassifier(max_features=21, max_depth=11, min_samples_split=4, n_estimators=136)
model.fit(X_train, y_train)

score = model.score(X_test, y_test)
print("test data accuracy=%.2f%%" % (score * 100))

score = model.score(X_train, y_train)
print("train data accuracy=%.2f%%" % (score * 100))


# In[122]:


from sklearn.metrics import confusion_matrix
y_pred_rf = model.predict(X_test)
confusion_matrix(y_test, y_pred, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])


# In[123]:


model = DecisionTreeClassifier(max_depth=15)
model.fit(X_train, y_train)
y_pred_dt = model.predict(X_test)


# In[126]:


imageData = None
for index, val in enumerate(y_test):
    if val != y_pred_dt[index] and val == y_pred_randomForest[index]:
        imageData = X_test[index]
        break
        
print("Correct detection in Random Forest and incorrect detection in Decision Tree")
plt.imshow(imageData.reshape((28,28)), cmap="Greys")


# # PCA
# 
# این الگوریتم یکی از الگوریتم های کاهش ابعاد است که با استفاده از واریانس هر ویژگی و روابط ریاضی و ويژگی های قبلی ویژگی های جدید را با تعداد کمتر یعنی ابعاد کمتر به وجود می آورد. برای فهم بهتر می توان این طور در نظر گرفت که ویژگی هایی خوب است که واریانس بیشتری داشته باشند تا در تشخیص نهایی عملکرد بهتری داشته باشد. اینکه چرا از کاهش ابعاد استفاده می کنیم دلایل زیادی دارد مثل بهبود در کارایی تشخیص- کاهش محاسبات -کاهش هزینه های پیچیدگی و دلایل دیگر

# In[127]:


from sklearn.decomposition import PCA

pca = PCA(n_components=12)
pca_results_train = pca.fit_transform(X_train)
pca_results_test = pca.transform(X_test)

