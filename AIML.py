import pandas as pd
path="C:\\Users\\bond\\Desktop\\batch4\\15_clientsubscription\\training_set_label.csv"
data=pd.read_csv(path)
print(data)
print()
print(data.info())
print(data.isnull().sum())
import sklearn
from sklearn.preprocessing import LabelEncoder
le_job=LabelEncoder()
data['job_n']=le_job.fit_transform(data['job'])
data['marital']=data['marital'].map({'married':0,'single':1,'divorced':2})
data['education']=data['education'].map({'tertiary':0,'secondary':1,'unknown':2,'primary':3})
data['default']=data['default'].map({'no':0,'yes':1})
data['housing']=data['housing'].map({'no':0,'yes':1})
data['loan']=data['loan'].map({'no':0,'yes':1})
data['contact']=data['contact'].map({'unknown':0,'cellular':1,'telephone':2})
data['month']=data['month'].map({'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12})
data['poutcome']=data['poutcome'].map({'unknown':0,'failure':1,'other':2,'success':3})
inputs=data.drop(['job','subscribe'],axis=1)
output=data['subscribe']
print(inputs)
print(output)
import sklearn
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(inputs,output,train_size=0.8)
print(x_train)
print(x_test)
print(y_train)
print(y_test)
import sklearn
from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier(n_neighbors=13)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print(y_pred)
print(y_test)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)
res=model.predict([[71,2,3,0,1729,0,0,1,17,11,977,3,-1,0,0,9]])
print(res)