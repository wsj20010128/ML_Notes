import numpy as np
from .metrics import r2_score


class SimpleLinearRegression:

    def __init__(self):
        """Initialize Simple Linear Regression Model"""
        self.a_ = None
        self.b_ = None

    def fit(self, x_train, y_train):
        """
        train Simple Linear Regression Model
        using x_train, y_train
        :param x_train:
        :param y_train:
        :return:
        """
        assert x_train.ndim == 1, \
            "Simple Linear Regressor can only solve single feature training data"
        assert len(x_train) == len(y_train), \
            "the size of x_train must be equal to the size of y_train"

        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)

        # use dot product (faster)
        num = (x_train - x_mean).dot(y_train - y_mean)
        d = (x_train - x_mean).dot(x_train - x_mean)

        self.a_ = num / d
        self.b_ = y_mean - self.a_ * x_mean

        return self

    def predict(self, x_predict):
        """
        given x_predict, a data set to be predicted
        return outcome vector
        :param x_predict:
        :return:
        """
        assert x_predict.ndim == 1, \
            "Simple Linear Regressor can only solve single feature training data"
        assert self.a_ is not None and self.b_ is not None, \
            "must fit before predict!"

        return np.array([self._predict(x) for x in x_predict])

    def _predict(self, x_single):
        """
        given x_single, a single data to be predicted
        return a predicting outcome
        :param x_single:
        :return:
        """
        return self.a_ * x_single + self.b_

    def score(self, x_test, y_test):
        """
        calculate accuracy
        according to x_test, y_test
        :param x_test:
        :param y_test:
        :return:
        """
        y_predict = self.predict(x_test)
        return r2_score(y_test, y_predict)

    def __repr__(self):
        return "SimpleLinearRegression()"
