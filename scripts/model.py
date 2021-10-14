#!/usr/bin/env python3

import sys

import pandas

import boost

############################################################
# helper functions #########################################
############################################################


def TODO(m):
    raise NotImplementedError(m)


############################################################
# main #####################################################
############################################################


def main():
    print("reading training data")
    training_data = pandas.read_csv("data/numerai_training_data.csv", header=0)
    features = [f for f in list(training_data) if "feature" in f]
    target = "target_bernie"

    print("reading validation data")
    tournament = pandas.read_csv("data/numerai_tournament_data.csv", header=0)
    validation = tournament[tournament["data_type"] == "validation"]

    # each model is a function mapping a feature data frame to a prediction
    models = []
    models.append(
        boost.build_linear_isotonic(training_data.sample(frac=0.5), features, target)
    )
    boost.check_logloss(validation, models[-1], target)

    model_stages = 10
    for _ in range(1, model_stages):
        m0 = models[-1]
        t0 = training_data.sample(frac=0.5)

        # TODO randomize this choice
        t1 = t0.assign(y0=m0)
        t1 = t1[t1.y0 < float(t1["y0"].median())]
        print("building next 1D model")
        m1 = boost.build_linear_isotonic(t1, features, target)

        print("building 2D model")
        m2 = boost.build_2d_isotonic(t0, m0, m1, target)
        boost.check_logloss(validation, m2, target)

        models.append(m2)

    return 0


############################################################
# startup handling #########################################
############################################################

if __name__ == "__main__":
    sys.exit(main())
