import numpy as np


def majorityVote(models_preds):
    final_pred = []

    models_preds_array = np.array(models_preds).T
    for row in models_preds_array:
        counts = np.bincount(row, minlength=2)
        predicted_class = np.argmax(counts)
        final_pred.append(predicted_class)

    return final_pred
