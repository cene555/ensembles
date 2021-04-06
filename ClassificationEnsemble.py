from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, precision_score, recall_score, roc_auc_score, jaccard_score
from collections import Counter
import numpy as np
import random
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier

class ClassificationEnsemble:
    def __init__(self, models, ensembling_type='bagging', bagging_type='majority_vote'):
        self.models = models
        if ensembling_type not in ['bagging', 'stacking']:
            raise NameError('There are only bagging, stacking ensembling types.')
        if bagging_type not in ['majority_vote', 'soft_vote']:
            raise NameError('There are only bagging, stacking bagging types.')
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
            predicts2 = np.array(predicts1).T
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
            predicts2 = np.array(predicts1).T
            return self.meta_model.predict_proba(predicts2)
    def compute_metrics(self, predicts, labels, metrics):
        if type(metrics) == type(' '):
            metrics = [metrics]
        results = {}
        for metric in metrics:
            if metric == 'accuracy_score':
                evaluation = accuracy_score(predicts, labels)
            elif metric == 'f1_score':
                evaluation = f1_score(predicts, labels)
            elif metric == 'balanced_accuracy_score':
                evaluation = balanced_accuracy_score(predicts, labels)
            elif metric == 'precision_score':
                evaluation = precision_score(predicts, labels)
            elif metric == 'recall_score':
                evaluation = recall_score(predicts, labels)
            elif metric == 'roc_auc_score':
                evaluation = roc_auc_score(predicts, labels)
            elif metric == 'jaccard_score':
                evaluation = jaccard_score(predicts, labels)         
            else:
                raise NameError('There are only metrics accuracy_score, f1_score, balanced_accuracy_score, precision_score, recall_score, roc_auc_score, jaccard_score')
            results[metric] = evaluation
        return results
    def eval(self, data, labels, metrics, is_print=True):
        predicts = self.predict(data)
        evaluations = self.compute_metrics(predicts, labels, metrics)
        if is_print:
            for key in evaluations.keys():
                print(key + ' = ' + str(evaluations[key]))
        return evaluations
    def fit(self, train_data, labels, eval_data=None, eval_labels=None, metrics='accuracy', size_of_bootstrap=0.7, size_of_train_for_stacking=0.7):
        if self.ensembling_type == 'bagging':
            for i in range(len(self.models)):
                random_state = random.randint(0,100000)
                bootstrap_X, _a, bootstrap_y, _b =  train_test_split(train_data, labels, test_size=1-size_of_bootstrap, random_state=random_state)
                self.models[i].fit(bootstrap_X, bootstrap_y)
        elif self.ensembling_type == 'stacking':
            predicts1 = []
            random_state = random.randint(0,100000)
            X1, X2, y1, y2 =  train_test_split(train_data, labels, test_size=1-size_of_train_for_stacking, random_state=random_state)
            for i in range(len(self.models)):
                self.models[i].fit(X1, y1)
                predicts1.append(self.models[i].predict(X2))
            predicts2 = np.array(predicts1).T
            self.meta_model = CatBoostClassifier(iterations=50,
                           learning_rate=0.1,
                           depth=3)
            self.meta_model.fit(predicts2, y2, verbose=False)
        if type(eval_data) != type(None) and type(eval_labels) != type(None):
            self.eval(eval_data, eval_labels, metrics)
