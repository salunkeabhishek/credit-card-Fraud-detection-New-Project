import datetime
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import date
import sklearn.exceptions
from pylab import rcParams
#import clearplot.plot_functions as pf
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.svm import OneClassSVM
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import IsolationForest
from imblearn.under_sampling import NearMiss 
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import plot_confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn import datasets, metrics, preprocessing, svm
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report,accuracy_score
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
import pickle



from sklearn.ensemble import RandomForestClassifier

data_train = pd.read_csv("fraudTrain.csv")
data_test = pd.read_csv("fraudTest.csv")

#data_train.info()
#print(data_train.shape,data_test.shape)
df_train= data_train.sample(frac = 0.1,random_state=1)
df_test= data_test.sample(frac = 0.05,random_state=1)
#print(df_train.shape,df_test.shape)
#df_train.head()
#df_train.isnull().sum()
#df_test.isnull().sum()
count_classes = pd.value_counts(df_train['is_fraud'], sort = True)

#transaction Informtion
fraud_train = df_train[df_train['is_fraud']==1]
normal_train = df_train[df_train['is_fraud']==0]
fraud_test = df_test[df_test['is_fraud']==1]
normal_test = df_test[df_test['is_fraud']==0]

#drop columns
def dropCol(data):
    col_to_drop = ['trans_date_trans_time','Unnamed: 0','cc_num','first','last','trans_num']
    res = data.drop(col_to_drop,axis = 1)
    return res
df_train = dropCol(df_train)
df_test = dropCol(df_test)
#print ( df_train.shape, df_test.shape)
columns = df_train.columns.tolist()
columns = [c for c in columns if c not in ["is_fraud"]]
X_train = df_train[columns]
Y_train = df_train['is_fraud']
X_test = df_test[columns]
Y_test = df_test['is_fraud']

def age_years(born):
    return 2021 - int(born[0:4])
X_train['age'] = X_train['dob'].apply(lambda x: age_years(x))
X_train = X_train.drop(['dob'],axis =1)
X_test['age'] = X_test['dob'].apply(lambda x: age_years(x))
X_test = X_test.drop(['dob'],axis =1)
#print(X_train.shape,X_test.shape) = (129668, 17) (27786, 17)

#one hot coding
# concanating the test and train data so that number of columns remain the same in both the data sets 
final_df = pd.concat([X_train,X_test],axis=0)
#final_df.shape = (157454, 16)

# creating the list of categorical variables
categorical_features =[feature for feature in X_train.columns if final_df[feature].dtypes == 'O']
#categorical_features = ['merchant', 'category', 'gender', 'street', 'city', 'state', 'job']
#observing the unique values in each feature
for feature in categorical_features:
    print("Distinct categories for {}  are {}".format(feature,len(final_df[feature].unique())))

#One Hot Coding
def category_onehot_multcols(data,multcolumns):
    df_final = data
    i=0
    for fields in multcolumns:
        print(fields)
        df1=pd.get_dummies(final_df[fields],drop_first=True)
        final_df.drop([fields],axis=1,inplace=True)
        if i==0:
            df_final=df1.copy()
        else:           
            df_final=pd.concat([df_final,df1],axis=1)
        i=i+1             
    df_final=pd.concat([final_df,df_final],axis=1)
    return df_final

final_df = category_onehot_multcols(final_df, categorical_features)
# removing duplicated columns
final_df =final_df.loc[:,~final_df.columns.duplicated()]
#final_df.shape
# separating the test and training data
df_Train=final_df.iloc[:129668,:]
df_Test=final_df.iloc[129668:,:]
#print(df_Train.shape,df_Test.shape) = (129668, 3077) (27786, 3077)

model_LR = LogisticRegression(random_state=137)

model_LR.fit(df_Train,Y_train)
y_pred = model_LR.predict(df_Test)

def print_eval(y_pred,model):
    print("Training Accuracy: ",model.score(df_Train, Y_train))
    print("Testing Accuracy: ", model.score(df_Test, Y_test))
    cm = confusion_matrix(Y_test, y_pred)
    print(cm)
    print(classification_report(Y_test,y_pred))

pickle.dump(model_LR, open('model1.pkl','wb'))
model=pickle.load(open('model1.pkl','rb'))