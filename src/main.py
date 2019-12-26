from bert_fine_tuning import BertClassifier
from constants import (
    DATA_FOLDER,
    MODEL_FOLDER,
    PREDICTION_FOLDER,
    PREDEFINED_COLUMNS,
)

import os
# import numpy as np
import pandas as pd


def checkModelCache():
    return len(os.listdir(MODEL_FOLDER)) > 0


def checkPredictionCache(filepath):
    return os.path.exists(filepath)


def fetch_model(target='tn', overwrite=False):
    num_labels = len(PREDEFINED_COLUMNS)
    data_filepath = os.path.join(
        DATA_FOLDER, '{TARGET}.csv'.format(TARGET=target))
    classifier = BertClassifier(num_labels=num_labels, debug_steps=11, progress=True)
    if not overwrite:
        if checkModelCache():
            print('Load cached model from disk')
            classifier.load_model()
        else:
            print('No cached model is found on disk; please set overwrite to true')
    else:
        df = pd.read_csv(data_filepath, delimiter=',')
        print('Start fitting model')
        classifier.fit(
            df.comment_text.values.tolist(),
            df[PREDEFINED_COLUMNS].values.tolist(),
        )
        print('Finish fitting model\n')
        classifier.save_model()
    return classifier


def fetch_predictions(classifier, target='tn', overwrite=False):
    pred_filepath = os.path.join(
        PREDICTION_FOLDER,
        '{TARGET}_pred.csv'.format(TARGET=target),
    )
    data_filepath = os.path.join(
        DATA_FOLDER,
        '{TARGET}.csv'.format(TARGET=target),
    )
    if not overwrite:
        if checkPredictionCache(pred_filepath):
            print('Load cached prediction form disk')
            return pd.read_csv(pred_filepath, delimiter=',').tolist()
        else:
            print('No cached prediction is found on disk; please set overwrite to true')
    else:
        df = pd.read_csv(data_filepath, delimiter=',')
        preds = classifier.predict_prob(
            df.comment_text.values.tolist(),
        )
        ndf = df[['id'] + PREDEFINED_COLUMNS].copy()
        ndf[PREDEFINED_COLUMNS] = preds
        ndf.to_csv(pred_filepath, index=False)
        return preds


def run():
    print('Start fetching model')
    classifier = fetch_model(target='tn', overwrite=False)
    print('Finish fetching model\n')

    print('Start label prediction')
    preds = fetch_predictions(classifier, target='tt', overwrite=True)
    print('Finish label prediction\n')


def main():
    run()


if __name__ == '__main__':
    main()
