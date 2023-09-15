from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier


from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler

import neurokit2 as nk
import numpy as np
import pandas as pd
import os
import re


def eda_custom_process(eda_signal, sampling_rate=4):
    eda_signal = nk.signal_sanitize(eda_signal)
    if type(eda_signal) is pd.Series and type(eda_signal.index) != pd.RangeIndex:
        eda_signal = eda_signal.reset_index(drop=True)
    eda_decomposed = nk.eda_phasic(eda_signal, sampling_rate=sampling_rate)
    return eda_decomposed["EDA_Phasic"]



class MachineLearningPipeline:
    def __init__(self, file_path):
        self.file_path = file_path
        try:
            self.data = pd.read_csv(file_path)
        except pd.errors.ParserError:
            self.data = pd.read_csv(file_path, engine='python')

    def extract_phasic_component(self):
        signal = self.data['MEAN_EDA_WRIST']
        self.data['EDA_Phasic'] = eda_custom_process(signal)

        #processed_signals, info = nk.eda_process(signal, sampling_rate=4)
        #self.data['EDA_Phasic'] = processed_signals["EDA_Phasic"]
        

    def preprocess_data(self):
        scaler = RobustScaler()
        self.data['STATE'] = self.data['STATE'] - 1
        self.data = self.data[self.data['STATE'].isin([0, 1])]
        self.data['EDA_Phasic'] = scaler.fit_transform(self.data[['EDA_Phasic']])
        #print(self.data)
        print(self.data['EDA_Phasic'].describe())
        print(self.data['STATE'].value_counts(normalize=True))

    def split_data(self):
        unique_subjects = self.data['SUBJECT'].unique()
        
        # Asegúrate de que haya al menos 3 sujetos únicos
        if len(unique_subjects) < 3:
            raise ValueError("No hay suficientes sujetos únicos para dividir los datos correctamente.")
        
        test_subjects = np.random.choice(unique_subjects, 3, replace=False)
        
        #self.X_train = self.data[~self.data['SUBJECT'].isin(test_subjects)][['MEAN_EDA_WRIST']]
        #self.y_train = self.data[~self.data['SUBJECT'].isin(test_subjects)]['STATE']
        
        #self.X_test = self.data[self.data['SUBJECT'].isin(test_subjects)][['MEAN_EDA_WRIST']]
        #self.y_test = self.data[self.data['SUBJECT'].isin(test_subjects)]['STATE']

        self.X_train = self.data[~self.data['SUBJECT'].isin(test_subjects)][['EDA_Phasic']]
        self.y_train = self.data[~self.data['SUBJECT'].isin(test_subjects)]['STATE']
        
        self.X_test = self.data[self.data['SUBJECT'].isin(test_subjects)][['EDA_Phasic']]
        self.y_test = self.data[self.data['SUBJECT'].isin(test_subjects)]['STATE']


    def train_and_evaluate(self, model, params):
        grid = GridSearchCV(model, params, cv=KFold(n_splits=5), scoring='accuracy',n_jobs=-1)
        grid.fit(self.X_train, self.y_train)
        
        y_pred = grid.best_estimator_.predict(self.X_test)
        
        # Calcula todas las métricas que necesitas
        accuracy = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        roc_auc = roc_auc_score(self.y_test, grid.best_estimator_.predict_proba(self.X_test)[:, 1])

        # Calcula las métricas ponderadas
        weighted_f1 = f1_score(self.y_test, y_pred, average='weighted')
        weighted_precision = precision_score(self.y_test, y_pred, average='weighted')
        weighted_recall = recall_score(self.y_test, y_pred, average='weighted')
        weighted_roc_auc = roc_auc_score(self.y_test, grid.best_estimator_.predict_proba(self.X_test)[:, 1], average='weighted')

        return model.__class__.__name__, accuracy, precision, recall, f1, weighted_f1, weighted_precision, weighted_recall, roc_auc, weighted_roc_auc


directory_path = "../data/data_wrist/"
file_paths = [os.path.join(directory_path, file_name) for file_name in os.listdir(directory_path) if file_name.endswith('.csv')]
def sorting_key(file_path):
    numbers = re.findall(r'\d+', os.path.basename(file_path))
    return int(numbers[0]), int(numbers[1])

# Ordenando los file_paths según los criterios definidos en la función sorting_key
file_paths.sort(key=sorting_key)

results_df = pd.DataFrame(columns=[
    "MODEL", "VENTANA_OVER", "ACCURACY", "PRECISION", "RECALL", "F1", 
    "WEIGHTED_PRECISION", "WEIGHTED_RECALL", "WEIGHTED_F1", "AUC_ROC", "WEIGHTED_AUC_ROC"
])

# Lista de los modelos a probar junto con un conjunto de hiperparámetros para cada modelo
models_and_params = [
    (LogisticRegression(), {'C': [0.01, 0.1, 1, 10, 100], 'penalty': ['l1', 'l2', 'elasticnet'], 'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}),
    (SVC(probability=True), {'C': [0.1, 1, 10, 100], 'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'gamma': ['scale', 'auto']}),
    (RandomForestClassifier(), {'n_estimators': [50, 100, 150], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}),
    (KNeighborsClassifier(), {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance'], 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}),
    (GradientBoostingClassifier(), {'n_estimators': [50, 100, 150], 'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 5, 10]}),
    (AdaBoostClassifier(), {'n_estimators': [50, 100, 150], 'learning_rate': [0.01, 0.1, 1]}),
    (XGBClassifier(eval_metric='logloss'), {'n_estimators': [50, 100, 150], 'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 5, 10]}),
    (LGBMClassifier(), {'n_estimators': [50, 100, 150], 'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 5, 10]}),
    (CatBoostClassifier(verbose=0), {'iterations': [50, 100, 150], 'learning_rate': [0.01, 0.1, 0.2], 'depth': [3, 5, 7, 9]}),
    (GaussianNB(), {}),
    (DecisionTreeClassifier(), {'criterion': ['gini', 'entropy'], 'splitter': ['best', 'random'], 'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}),
]

best_model_info = {'Model': '', 'File': '', 'WEIGHTED_F1': 0}

# Bucle sobre cada archivo y cada modelo
for file_path in file_paths:
    print(f"Processing file: {file_path}\n")
    pipeline = MachineLearningPipeline(file_path)
    pipeline.extract_phasic_component()
    pipeline.preprocess_data()
    pipeline.split_data()
    
    for model, params in models_and_params:
        model_name, accuracy, precision, recall, f1, weighted_f1, weighted_precision, weighted_recall, roc_auc, weighted_roc_auc = pipeline.train_and_evaluate(model, params)

        new_row = pd.Series({
            "MODEL": model_name,
            "VENTANA_OVER": file_path,
            "ACCURACY": accuracy,
            "PRECISION": precision,
            "RECALL": recall,
            "F1": f1,
            "WEIGHTED_PRECISION": weighted_precision,
            "WEIGHTED_RECALL": weighted_recall,
            "WEIGHTED_F1": weighted_f1,
            "AUC_ROC": roc_auc,
            "WEIGHTED_AUC_ROC": weighted_roc_auc
        })
        results_df = pd.concat([results_df, new_row.to_frame().T], ignore_index=True)
        print(f"Model: {model_name}, Accuracy: {accuracy:.5f}, WEIGHTED_F1: {weighted_f1:.5f}")
        if weighted_f1 > best_model_info['WEIGHTED_F1']:
            best_model_info.update({'Model': model_name, 'File': os.path.basename(file_path), 'WEIGHTED_F1': weighted_f1})

results_df.iloc[:, 2:] = results_df.iloc[:, 2:].astype(float).apply(lambda x: round(x, 5))
results_df.to_csv("model_results.csv", index=False)
print(f"\nBest model based on WEIGHTED_F1: {best_model_info['Model']} from file: {best_model_info['File']}\nWEIGHTED_F1: {best_model_info['WEIGHTED_F1']:.5f}")