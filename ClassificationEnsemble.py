from sklearn.metrics import accuracy_score, f1_score
from collections import Counter
import numpy as np
import random
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier

class ClassificationEnsemble:
    def __init__(self, models, ensembling_type='bagging', bagging_type='majority_vote'):
        self.models = models
        self.ensembling_type = ensembling_type
        self.bagging_type=bagging_type
    def predict(self, data):
        if self.ensembling_type == 'bagging':
            if self.bagging_type == 'majority_vote':
                predicts = []
                for i in range(len(self.models)):
                    predict = self.models[i].predict(data)
                    predicts.append(predict)
                final_predicts = []
                for i in range(len(predicts[0])):
                    preds = []
                    for j in range(len(predicts)):
                        preds.append(predicts[j][i])
                    final_predicts.append(Counter(preds).most_common()[0][0])
                return final_predicts
            elif self.bagging_type == 'soft_vote':
                predicts = np.array([])
                for i in range(len(self.models)):
                    if i == 0:
                        predicts = self.models[i].predict_proba(data)
                    else:
                        predicts += self.models[i].predict_proba(data)
                return np.argmax(predicts / len(self.models), axis=1)
        elif self.ensembling_type == 'stacking':
            predicts1 = []
            for i in range(len(self.models)):
                predicts1.append(self.models[i].predict(data))
            predicts2 = []
            for i in range(len(predicts1[0])):
                predicts2.append([])
                for j in range(len(predicts1)):
                    predicts2[i].append(predicts1[j][i])
            return self.meta_model.predict(predicts2)
    def predict_proba(self, data):
        if self.ensembling_type == 'bagging':
            predicts = np.array([])
            for i in range(len(self.models)):
                if i == 0:
                    predicts = self.models[i].predict_proba(data)
                else:
                    predicts += self.models[i].predict_proba(data)
            return predicts / len(self.models)
        elif self.ensembling_type == 'stacking':
            predicts1 = []
            for i in range(len(self.models)):
                predicts1.append(self.models[i].predict(data))
            predicts2 = []
            for i in range(len(predicts1[0])):
                predicts2.append([])
                for j in range(len(predicts1)):
                    predicts2[i].append(predicts1[j][i])
            return self.meta_model.predict_proba(predicts2)
          
    
    def fit(self, train_data, labels, eval_data=None, eval_labels=None, metric='accuracy'):
        if self.ensembling_type == 'bagging':
            for i in range(len(self.models)):
                random_state = random.randint(0,100000)
                bootstrap_X, _a, bootstrap_y, _b =  train_test_split(train_data, labels, test_size=0.3, random_state=random_state)
                self.models[i].fit(bootstrap_X, bootstrap_y)
            if type(eval_data) != type(None) and type(eval_labels) != type(None):
                predicts = self.predict(eval_data)
                if metric == 'accuracy':
                    print(accuracy_score(eval_labels, predicts))
                elif metric == 'f1':
                    print(f1_score(eval_labels, predicts))
        elif self.ensembling_type == 'stacking':
            predicts1 = []
            random_state = random.randint(0,100000)
            X1, X2, y1, y2 =  train_test_split(train_data, labels, test_size=0.5, random_state=random_state)
            for i in range(len(self.models)):
                self.models[i].fit(X1, y1)
                predicts1.append(self.models[i].predict(X2))
            predicts2 = []
            for i in range(len(predicts1[0])):
                predicts2.append([])
                for j in range(len(predicts1)):
                    predicts2[i].append(predicts1[j][i])
            self.meta_model = CatBoostClassifier(iterations=10,
                           learning_rate=0.1,
                           depth=1)
            self.meta_model.fit(predicts2, y2)
            if type(eval_data) != type(None) and type(eval_labels) != type(None):
                predicts = self.predict(eval_data)
                if metric == 'accuracy':
                    print(accuracy_score(eval_labels, predicts))
                elif metric == 'f1':
                    print(f1_score(eval_labels, predicts))
