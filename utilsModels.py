import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
from utils.visualization import plt_fig


def calculScoreAndVisualize(model, scores, gt_list, gt_mask_list, test_imgs):
    max_anomaly_score = scores.max()
    min_anomaly_score = scores.min()
    scores = (scores - min_anomaly_score) / \
        (max_anomaly_score - min_anomaly_score)

    img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
    gt_list = np.asarray(gt_list)

    img_roc_auc = roc_auc_score(gt_list, img_scores)
    print('image ROCAUC: %.3f' % (img_roc_auc))
    if model.vis:
        precision, recall, thresholds = precision_recall_curve(
            gt_list.flatten(), img_scores.flatten())
        a = 2 * precision * recall
        b = precision + recall
        desired_precision = 0.8
        idx = next(i for i, pr in enumerate(
            precision) if pr >= desired_precision)

        desired_precision = precision[idx]
        desired_recall = recall[idx]
        print("precision : "+str(desired_precision))
        print("recall : " + str(desired_recall))
        f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
        cls_threshold = thresholds[np.argmax(f1)]

        gt_mask = np.asarray(gt_mask_list)
        gt_mask = gt_mask.astype(int)
        precision, recall, thresholds = precision_recall_curve(
            gt_mask.flatten(), scores.flatten())
        a = 2 * precision * recall
        b = precision + recall
        f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
        seg_threshold = thresholds[np.argmax(f1)]
        print("segmentation threshold : " + str(seg_threshold))
        print(model.obj)

        plt_fig(test_imgs, scores, img_scores, gt_mask_list, seg_threshold, cls_threshold,
                model.img_dir, model.obj)
