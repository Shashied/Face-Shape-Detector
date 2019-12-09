import pickle
train_data=pickle.load(open('data.pickle','rb'))
train_target=pickle.load(open('target.pickle','rb'))
#loading the arrays saved in last code


print(train_data.shape)
print(train_target.shape)

from sklearn.neighbors import KNeighborsClassifier

clsfr=KNeighborsClassifier()
clsfr.fit(train_data,train_target)
#training KNN

import joblib

joblib.dump(clsfr,'KNN_model.sav')
