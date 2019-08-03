import numpy as np
from scipy import ndimage as ndi
from skimage.exposure import histogram

__all__ = ['batch_intersection_union',
           'batch_jaccard_index_and_dice_coefficient',
           'batch_pix_accuracy',
           'batch_precision_recall',
           'f1_score',
           'batch_sens_spec']


def gaussian_threshold(image, block_size=100, offset=-0.06, cval=0):
    # automatically determine sigma which covers > 99% of distribution
    thresh_image = np.zeros(image.shape, 'double')
    sigma = (block_size - 1) / 6.0
    ndi.gaussian_filter(image, sigma, output=thresh_image, mode='reflect', cval=cval)
    return thresh_image - offset


def otsu_threshold(image, nbins=256):
    hist, bin_centers = histogram(image.ravel(), nbins)
    hist = hist.astype(float)

    # class probabilities for all possible thresholds
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]
    # class means for all possible thresholds
    mean1 = np.cumsum(hist * bin_centers) / weight1
    mean2 = (np.cumsum((hist * bin_centers)[::-1]) / weight2[::-1])[::-1]

    # Clip ends to align class 1 and class 2 variables:
    # The last value of `weight1`/`mean1` should pair with zero values in
    # `weight2`/`mean2`, which do not exist.
    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

    idx = np.argmax(variance12)
    threshold = bin_centers[:-1][idx]
    return threshold


def batch_pix_accuracy(predict, target, thr=0.9):
    predict = predict > thr
    predict = predict.data.cpu().numpy() + 1
    target = target.data.cpu().numpy() + 1
    pixel_labeled = np.sum(target > 0)
    pixel_correct = np.sum((predict == target) * (target > 0))
    assert pixel_correct <= pixel_labeled, \
        "Correct area should be smaller than Labeled"
    acc = np.float64(1.0) * pixel_correct / (np.spacing(1, dtype=np.float64) + pixel_labeled)
    return acc


def batch_intersection_union(predict, target, nclass=2, thr=0.9):
    predict = predict > thr
    mini = 1
    maxi = nclass
    nbins = nclass
    predict = predict.data.cpu().numpy() + 1
    target = target.data.cpu().numpy() + 1

    predict = predict * (target > 0).astype(predict.dtype)
    intersection = predict * (predict == target)
    # areas of intersection and union
    area_inter, _ = np.histogram(intersection, bins=nbins, range=(mini, maxi))
    area_pred, _ = np.histogram(predict, bins=nbins, range=(mini, maxi))
    area_lab, _ = np.histogram(target, bins=nbins, range=(mini, maxi))
    area_union = area_pred + area_lab - area_inter
    assert (area_inter <= area_union).all(), \
        "Intersection area should be smaller than Union area"
    IoU = (np.float64(1.0) * area_inter / (np.spacing(1, dtype=np.float64) + area_union)).mean()

    return IoU


def batch_precision_recall(predict, target, thr=0.9):
    predict = predict > thr
    predict = predict.data.cpu().numpy() + 1
    target = target.data.cpu().numpy() + 1

    tp = np.sum(((predict == 2) * (target == 2)) * (target > 0))
    fp = np.sum(((predict == 2) * (target == 1)) * (target > 0))
    fn = np.sum(((predict == 1) * (target == 2)) * (target > 0))

    precision = float(np.nan_to_num(tp / (tp + fp)))
    recall = float(np.nan_to_num(tp / (tp + fn)))

    return precision, recall


def batch_sens_spec(predict, target, thr=0.9):
    predict = predict > thr
    predict = predict.data.cpu().numpy() + 1
    target = target.data.cpu().numpy() + 1

    tp = np.sum(((predict == 2) * (target == 2)) * (target > 0))
    fp = np.sum(((predict == 2) * (target == 1)) * (target > 0))
    tn = np.sum(((predict == 1) * (target == 1)) * (target > 0))
    fn = np.sum(((predict == 1) * (target == 2)) * (target > 0))

    sensitivity = float(np.nan_to_num(tp / (tp + fn)))
    specificity = float(np.nan_to_num(tn / (tn + fp)))

    return sensitivity, specificity


def batch_jaccard_index_and_dice_coefficient(predict, target, thr=0.9):
    predict = predict > thr
    predict = predict.data.cpu().numpy() + 1
    target = target.data.cpu().numpy() + 1

    tp = np.sum(((predict == 2) * (target == 2)) * (target > 0))
    fp = np.sum(((predict == 2) * (target == 1)) * (target > 0))
    fn = np.sum(((predict == 1) * (target == 2)) * (target > 0))

    ji = float(np.nan_to_num(tp / (tp + fp + fn)))
    dice = float(np.nan_to_num(2 * tp / (2 * tp + fp + fn)))

    return ji, dice


def f1_score(predict, target, thr=0.9):
    f1 = 0
    precision, recall = batch_precision_recall(predict, target, thr)
    if precision + recall > 0:
        f1 = 2 * np.nan_to_num((precision * recall) / (precision + recall))
    return f1
