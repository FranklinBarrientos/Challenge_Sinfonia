import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import linear_model

import shap

class ML_LogisticRegression:

    def __init__(self,df):

        self.data_final = df.dropna()

    def Prediction(self):

        data_activo = self.data_final[self.data_final['Estado del beneficiarios'] == 1]
        data_inactivo = self.data_final[self.data_final['Estado del beneficiarios'] == 0]
        X = data_activo.iloc[:,1:]
        y = data_activo['Estado del beneficiarios'].values

        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=1)

        X_train['Estado del beneficiarios'] = y_train
        data_train = pd.concat([data_inactivo,X_train]).sample(frac = 1)
        X_train = data_train.iloc[:,1:]
        y_train = data_train['Estado del beneficiarios'].values

        columns_shap = list(X.columns)

        scaler = StandardScaler()
        X_train_std = scaler.fit(X_train).transform(X_train)
        X_test_std = scaler.transform(X_test)

        log_cl = linear_model.LogisticRegression().fit(X_train_std, y_train)
        
        y_predict = log_cl.predict(X_test_std)
        logistic_accuracy = log_cl.score(X_test_std,y_test)

        explainer_log = shap.Explainer(log_cl, X_train_std, feature_names=columns_shap)
        
        shap_values_log = explainer_log(X_train_std)

        X_train['Estado del beneficiarios'] = y_train
        X_train['Estado del beneficiarios predict'] = log_cl.predict(X_train_std)
        X_train[['Prediction Probability of 0','Prediction Probability of 1']] = log_cl.predict_proba(X_train_std)

        X_test['Estado del beneficiarios'] = y_test
        X_test['Estado del beneficiarios predict'] = log_cl.predict(X_test_std)
        X_test[['Prediction Probability of 0', 'Prediction Probability of 1']] = log_cl.predict_proba(X_test_std)

        return (X_train, X_test, y_predict, logistic_accuracy, shap_values_log)