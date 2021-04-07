from sklearn.metrics import explained_variance_score, max_error, mean_squared_error, mean_absolute_error, mean_squared_log_error, median_absolute_error, r2_score, mean_poisson_deviance, mean_gamma_deviance
from collections import Counter
import numpy as np
import random
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor

class RegressorEnsemble:
    def __init__(self, models, ensembling_type='bagging'):
        self.models = models
        if ensembling_type not in ['bagging', 'stacking']:
            raise NameError('There are only bagging, stacking ensembling types.')
        self.ensembling_type = ensembling_type
    def predict(self, data):
        if self.ensembling_type == 'bagging':
            predicts = np.array([])
            for i in range(len(self.models)):
                if i == 0:
                    predicts = np.array(self.models[i].predict(data))
                else:
                    predicts += np.array(self.models[i].predict(data))
            return predicts / len(self.models)
        elif self.ensembling_type == 'stacking':
            predicts1 = []
            for i in range(len(self.models)):
                predicts1.append(self.models[i].predict(data))
            predicts2 = np.array(predicts1).T
            return self.meta_model.predict(predicts2)
    def compute_metrics(self, predicts, labels, metrics):
        if type(metrics) == type(' '):
            metrics = [metrics]
        results = {}
        for metric in metrics:
            if metric == 'explained_variance_score':
                evaluation = explained_variance_score(predicts, labels)
            elif metric == 'max_error':
                evaluation = max_error(predicts, labels)
            elif metric == 'mean_squared_error':
                evaluation = mean_squared_error(predicts, labels)
            elif metric == 'mean_absolute_error':
                evaluation = mean_absolute_error(predicts, labels)
            elif metric == 'mean_squared_log_error':
                evaluation = mean_squared_log_error(predicts, labels)
            elif metric == 'median_absolute_error':
                evaluation = median_absolute_error(predicts, labels)
            elif metric == 'r2_score':
                evaluation = r2_score(predicts, labels)
            elif metric == 'mean_poisson_deviance':
                evaluation = mean_poisson_deviance(predicts, labels)      
            elif metric == 'mean_gamma_deviance':
                evaluation = mean_gamma_deviance(predicts, labels)               
            else:
                raise NameError('There are only metrics explained_variance_score, max_error, mean_squared_error, mean_absolute_error, mean_squared_log_error, median_absolute_error, r2_score, mean_poisson_deviance, mean_gamma_deviance')
            results[metric] = evaluation
        return results
    def eval(self, data, labels, metrics, is_print=True):
        predicts = self.predict(data)
        evaluations = self.compute_metrics(predicts, labels, metrics)
        if is_print:
            for key in evaluations.keys():
                print(key + ' = ' + str(evaluations[key]))
        return evaluations
    def fit(self, train_data, labels, eval_data=None, eval_labels=None, metrics='mean_squared_error', size_of_bootstrap=0.7, size_of_train_for_stacking=0.7):
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
            self.meta_model = CatBoostRegressor(iterations=50,
                           learning_rate=0.1,
                           depth=3)
            self.meta_model.fit(predicts2, y2, verbose=False)
        if type(eval_data) != type(None) and type(eval_labels) != type(None):
            self.eval(eval_data, eval_labels, metrics)
