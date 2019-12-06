import os
import logging
import numpy as np


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

    def calculate_scores(self, score_types=None):
        if score_types is None:
            score_types = ['bbox']

        for score_type in score_types:
            if score_type == 'bbox':
                self.bbox_score()
            elif score_type == 'mask':
                self.mask_score()
            else:
                raise Exception('{} score type not supported'.format(score_type))

    def bbox_score(self):

        for targets, predictions in self.data:
            bboxes_targets = targets['boxes']
            bboxes_t_areas = targets['area']

            bboxes_pred = predictions['boxes']
            bboxes_pred_score = predictions['scores']

            # apply non-max suppression to predictions
            bboxes_pred_suppr = self.non_max_suppr_binary(bboxes_pred_score, bboxes_pred)

            # compute bounding box scores in comparison with

    def mask_score(self):
        pass

    def non_max_suppr_binary(self, bboxes_pred_score, bboxes_pred, base_threshold=0.6):
        # binary classification version of non-max suppression

        # firstly we discard all bbox predictions where class prob < base_treshold
        selected_idx = np.argwhere(bboxes_pred_score > base_threshold).flatten()
        selected_bboxes = bboxes_pred[selected_idx]
        selected_scores = bboxes_pred_score[selected_idx]

        out = []
        # continue iterations until the list of scores is depleted
        while len(selected_scores) > 0:
            highest_score_idx = np.argmax(selected_scores)
            selected_scores = np.delete(selected_scores, highest_score_idx)

            top_bbox = selected_bboxes[highest_score_idx]
            selected_bboxes = np.delete(selected_bboxes, highest_score_idx, axis=0)

            if len(selected_bboxes.shape) == 1:
                selected_bboxes = np.expand_dims(selected_bboxes, 0)

            # if we pick the last item from selected_scores and boxes, add it directly to the results
            # since there are no items left to compare it against
            if len(selected_scores) > 0:
                duplicate_boxes_idx = []
                for idx, remain_box in enumerate(selected_bboxes):
                    iou = self.intersection_over_union(top_bbox, remain_box)
                    if iou > 0.5:
                        duplicate_boxes_idx.append(idx)
                # drop duplicate boxes with high intersection if any are found
                selected_scores = np.delete(selected_scores, duplicate_boxes_idx)
                selected_bboxes = np.delete(selected_bboxes, duplicate_boxes_idx, axis=0)

            out.append(top_bbox)

        # stack the collected results
        res = np.stack(out)
        return res

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
