import numpy as np

from sklearn.metrics import accuracy_score, classification_report, jaccard_score

def segmentation_accuracy(ground_truth, predicted):
    """
    Calculates the percentage of pixels that are the same in the ground truth and predicted images.

    ground_truth: 2D numpy array of 0s and 1s
    predicted: 2D numpy array of 0s and 1s
    """
    flattened_ground_truth = ground_truth.flatten()
    flattened_predicted = predicted.flatten()
    accuracy = accuracy_score(flattened_ground_truth, flattened_predicted)
    return accuracy

def segmentation_classification_report(ground_truth, predicted):
    """
    Calculates the precision, recall, and F1 score for the predicted segmentation.

    ground_truth: 2D numpy array of 0s and 1s
    predicted: 2D numpy array of 0s and 1s
    """
    flattened_ground_truth = ground_truth.flatten()
    flattened_predicted = predicted.flatten()
    return classification_report(flattened_ground_truth, flattened_predicted, labels=[0, 1], target_names=['background', 'foreground'], output_dict=True)

def segmentation_dice_score(ground_truth, predicted):
    """
    Calculates the Dice score for the predicted segmentation.

    ground_truth: 2D numpy array of 0s and 1s
    predicted: 2D numpy array of 0s and 1s
    """
    intersection = np.logical_and(ground_truth, predicted)
    dice_score = (2*np.sum(intersection))/(np.sum(ground_truth)+np.sum(predicted))
    return dice_score

def segmentation_jaccard_score(ground_truth, predicted):
    """
    Calculates the Jaccard score for the predicted segmentation.

    ground_truth: 2D numpy array of 0s and 1s
    predicted: 2D numpy array of 0s and 1s
    """
    flattened_ground_truth = ground_truth.flatten()
    flattened_predicted = predicted.flatten()

    jaccard = jaccard_score(flattened_ground_truth, flattened_predicted)
    return jaccard

def segmentation_metrics(ground_truth, predicted):
    """
    Returns a dictionary of segmentation metrics.

    ground_truth: 2D numpy array of 0s and 1s
    predicted: 2D numpy array of 0s and 1s
    """
    classification_report = segmentation_classification_report(ground_truth, predicted)
    dice_score = segmentation_dice_score(ground_truth, predicted)
    jaccard_score = segmentation_jaccard_score(ground_truth, predicted)

    return {
        "accuracy": classification_report["accuracy"],

        "precision_foreground": classification_report["foreground"]["precision"],
        "recall_foreground": classification_report["foreground"]["recall"],
        "f1_foreground": classification_report["foreground"]["f1-score"],
        "precision_background": classification_report["background"]["precision"],
        "recall_background": classification_report["background"]["recall"],
        "f1_background": classification_report["background"]["f1-score"],

        "dice_score": dice_score,
        "jaccard_score": jaccard_score,
    }

def average_dict_values(dicts):
    """
    Returns a dictionary where the values are the average of the values in the input dictionaries.

    dicts: list of dictionaries
    """
    keys = dicts[0].keys()
    averages = {}
    for key in keys:
        values = [d[key] for d in dicts]
        average = sum(values) / len(values)
        averages[key] = average
    return averages