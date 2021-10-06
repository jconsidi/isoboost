# jcboost.py

"""
My homebrew boosting library.
"""

import numpy
import sklearn.isotonic
import sklearn.linear_model

from isoboost import *

def build_2d_isotonic(training_data, m0, m1, target):
    #if len(m0values) * len(m1values) > 10000:
    #    raise RuntimeError('too many buckets!')

    # append the two input models to the training data set

    training_data = training_data.assign(m0 = m0, m1 = m1)
    training_data = training_data[['m0', 'm1', target]]

    model_input = list(training_data.itertuples())
    xs = [r[0] for r in model_input]
    ys = [r[1] for r in model_input]
    vs = [r[2] for r in model_input]

    model = isotonic2d.regress_isotonic_2d(xs, ys, vs)

    return model

def build_linear(training_data, features, target):
    X = training_data[features]
    Y = training_data[target]

    linear_model = sklearn.linear_model.LinearRegression(copy_X = True, fit_intercept = True)
    linear_model.fit(X, Y)
    print('linear train score = %.4f' % (linear_model.score(X, Y)))

    return linear_model.predict

def build_linear_isotonic(training_data, features, target):
    X = training_data[features]
    Y = training_data[target]

    linear = sklearn.linear_model.LinearRegression(copy_X = True, fit_intercept = True)
    linear.fit(X, Y)
    Y_linear = linear.predict(X)

    isotonic = sklearn.isotonic.IsotonicRegression(out_of_bounds = 'clip')
    isotonic.fit_transform(Y_linear, Y)
    
    def model(X_in):
        l = linear.predict(X_in[features])
        i = isotonic.predict(l)
        return i

    return model

def check_logloss(validation_data, m, target):
    Y_target = validation_data[target]
    Y_model = m(validation_data)

    print('log loss = %.4f' % sklearn.metrics.log_loss(y_true = Y_target, y_pred = Y_model))
