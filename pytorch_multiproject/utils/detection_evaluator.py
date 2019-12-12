import os
import logging
import numpy as np
from random import sample


class DetectionEvaluator:

    def __init__(self):
        """
           To calculate the object detection scores, we accumulate ground truth labels (targets) and
           predictions of our model during the val phase, then compute the necessary metrics (e.g. bounding box mAP)
        """
        self.logger = logging.getLogger(os.path.basename(__file__))
        self.data = []

    def accumulate(self, targets, predictions):
        """
        targets (dict):
        predictions (dict):
        """
        self.data.append([targets, predictions])

    def bbox_score(self, iou_threshold=0.5, non_max_iou_thresh=0.5, score_threshold=0.6):

        true_positive = np.array([])
        false_positive = np.array([])
        num_ground_truths = 0

        for targets, predictions in self.data:
            bboxes_targets = targets['boxes']
            if not isinstance(bboxes_targets, np.ndarray):
                bboxes_targets = bboxes_targets.numpy()

            bboxes_pred = predictions['boxes']
            bboxes_pred_score = predictions['scores']

            # apply non-max suppression to predictions
            bboxes_pred_suppr, bbox_pred_scores_suppr = self.non_max_suppr_binary(bboxes_pred_score, bboxes_pred,
                                                                                  score_threshold, non_max_iou_thresh)

            # since number of predicted boxes is usually different from the number of true boxes, we need
            # to create all the possible combinations of true and predicted bbox coordinates for iou calculations
            targets_predictions_comb = np.hstack([np.repeat(bboxes_pred_suppr, bboxes_targets.shape[0], axis=0),
                                                  np.tile(bboxes_targets, (bboxes_pred_suppr.shape[0], 1))])

            self.logger.debug(targets_predictions_comb)
            # compute ious for all the possible combinations of predcitions and targets
            iou = self.batch_iou(targets_predictions_comb[:, :4], targets_predictions_comb[:, 4:])

            self.logger.debug(iou)
            # rearrange iou into separate groups - one group for each prediction
            # corresponding to ious of each prediction with all the ground truth (or target) bboxes
            iou = np.hsplit(iou, bboxes_pred_suppr.shape[0])
            self.logger.debug(iou)

            # intermediate containers to accumulate true and false positives during one iteration
            # note that length of one container corresponds to the number of predictions
            true_pos_iter = np.zeros(len(iou))
            false_pos_iter = np.zeros(len(iou))

            # collect the number of ground truths in each target - prediction pair for recall calculation
            num_ground_truths += bboxes_targets.shape[0]

            # iterate trough groups in calculated ious. One group corresponds to ious of one prediction with all
            # the targets.
            guessed_bboxes = []
            for group_idx, iou_group in enumerate(iou):
                for target_idx, iou in enumerate(iou_group):
                    if iou > iou_threshold and target_idx not in guessed_bboxes:
                        guessed_bboxes.append(target_idx)
                        true_pos_iter[group_idx] += 1
                    else:
                        false_pos_iter[group_idx] += 1

            self.logger.debug('guessed bboxes: ' + str(guessed_bboxes))
            true_positive = np.append(true_positive, true_pos_iter)
            false_positive = np.append(false_positive, false_pos_iter)

            self.logger.debug('collected_tps:'+str(true_positive))
            self.logger.debug('collected_fps:' + str(false_positive))

        accum_tp = np.cumsum(true_positive)
        accum_fp = np.cumsum(false_positive)

        precision = np.divide(accum_tp, (accum_tp + accum_fp))
        recall = accum_tp / num_ground_truths
        self.logger.debug('Precision :'+str(precision))
        self.logger.debug('Recall :'+str(recall))

        avg_precision, prec_interp, m_recall = self.get_average_precision(precision, recall)

        self.logger.debug('\n\n')
        return avg_precision, prec_interp, m_recall, precision, recall

    def mask_score(self):
        pass

    def non_max_suppr_binary(self, bboxes_pred_score, bboxes_pred, score_threshold, iou_threshold):
        # binary classification version of non-max suppression

        # firstly we discard all bbox predictions where class prob < base_treshold
        selected_idx = np.argwhere(bboxes_pred_score > score_threshold).flatten()
        selected_bboxes = bboxes_pred[selected_idx]
        selected_scores = bboxes_pred_score[selected_idx]

        out_bboxes = []
        out_scores = []
        # continue iterations until the list of scores is depleted
        while len(selected_scores) > 0:
            highest_score_idx = np.argmax(selected_scores)

            top_score = selected_scores[highest_score_idx]
            top_bbox = selected_bboxes[highest_score_idx]

            selected_scores = np.delete(selected_scores, highest_score_idx)
            selected_bboxes = np.delete(selected_bboxes, highest_score_idx, axis=0)

            # to prevent selected_bboxes matrix from collapsing into a vector
            if len(selected_bboxes.shape) == 1:
                selected_bboxes = np.expand_dims(selected_bboxes, 0)

            # if we pick the last item from selected_scores and boxes, add it directly to the results
            # since there are no items left to compare it against
            if len(selected_scores) > 0:
                duplicate_boxes_idx = []

                for idx, remain_box in enumerate(selected_bboxes):
                    iou = self.intersection_over_union(top_bbox, remain_box)
                    if iou > iou_threshold:
                        duplicate_boxes_idx.append(idx)

                # drop duplicate boxes with high intersection if any are found
                selected_scores = np.delete(selected_scores, duplicate_boxes_idx)
                selected_bboxes = np.delete(selected_bboxes, duplicate_boxes_idx, axis=0)

            out_bboxes.append(top_bbox)
            out_scores.append(top_score)

        # stack the collected results
        res_bbox = np.stack(out_bboxes)
        res_scores = np.stack(out_scores)
        return res_bbox, res_scores

    @staticmethod
    def get_average_precision(precision, recall):

        m_precision = list()
        m_precision.append(0)
        [m_precision.append(value) for value in precision]
        m_precision.append(0)

        m_recall = list()
        m_recall.append(0)
        [m_recall.append(value) for value in recall]
        m_recall.append(1)

        # interpolate precision by going backwards and overwriting all encountered precision values
        # with the largest found value
        for i in range(len(m_precision) - 1, 0, -1):
            m_precision[i - 1] = max(m_precision[i - 1], m_precision[i])

        # locate indices of steps in recall value list (places where recall values change)
        recall_deltas_idx = []
        for i in range(len(m_recall) - 1):
            if m_recall[1:][i] != m_recall[0:-1][i]:
                recall_deltas_idx.append(i + 1)

        # compute avg precision as an area of interpolated precision - recall squares
        avg_precision = 0
        for i in recall_deltas_idx:
            avg_precision = avg_precision + (m_recall[i] - m_recall[i - 1]) * m_precision[i]

        return avg_precision, m_precision, m_recall

    @staticmethod
    def intersection_over_union(bbox_1, bbox_2):
        bbox_1_x0 = bbox_1[0]
        bbox_1_y0 = bbox_1[1]
        bbox_1_x1 = bbox_1[2]
        bbox_1_y1 = bbox_1[3]

        bbox_2_x0 = bbox_2[0]
        bbox_2_y0 = bbox_2[1]
        bbox_2_x1 = bbox_2[2]
        bbox_2_y1 = bbox_2[3]

        # determine the coordinates of the intersection rectangle
        x_left = max(bbox_1_x0, bbox_2_x0)
        y_top = max(bbox_1_y0, bbox_2_y0)
        x_right = min(bbox_1_x1, bbox_2_x1)
        y_bottom = min(bbox_1_y1, bbox_2_y1)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersect_area = (x_right - x_left + 1) * (y_bottom - y_top + 1)
        bbox_1_area = (bbox_1_x1 - bbox_1_x0 + 1) * (bbox_1_y1 - bbox_1_y0 + 1)
        bbox_2_area = (bbox_2_x1 - bbox_2_x0 + 1) * (bbox_2_y1 - bbox_2_y0 + 1)

        iou = intersect_area / (bbox_1_area + bbox_2_area - intersect_area)
        return iou

    @staticmethod
    def batch_iou(bbox_array_1, bbox_array_2):
        bbox_1_x0 = bbox_array_1[:, 0]
        bbox_1_y0 = bbox_array_1[:, 1]
        bbox_1_x1 = bbox_array_1[:, 2]
        bbox_1_y1 = bbox_array_1[:, 3]

        bbox_2_x0 = bbox_array_2[:, 0]
        bbox_2_y0 = bbox_array_2[:, 1]
        bbox_2_x1 = bbox_array_2[:, 2]
        bbox_2_y1 = bbox_array_2[:, 3]

        # determine the coordinates of the intersection rectangle
        x_left = np.maximum(bbox_1_x0, bbox_2_x0)
        y_top = np.maximum(bbox_1_y0, bbox_2_y0)
        x_right = np.minimum(bbox_1_x1, bbox_2_x1)
        y_bottom = np.minimum(bbox_1_y1, bbox_2_y1)

        width = x_right - x_left + 1
        height = y_bottom - y_top + 1

        width[width < 0] = 0
        height[height < 0] = 0

        intersect_area = height * width
        bbox_1_area = (bbox_1_x1 - bbox_1_x0 + 1) * (bbox_1_y1 - bbox_1_y0 + 1)
        bbox_2_area = (bbox_2_x1 - bbox_2_x0 + 1) * (bbox_2_y1 - bbox_2_y0 + 1)

        iou = intersect_area / (bbox_1_area + bbox_2_area - intersect_area)
        return iou
