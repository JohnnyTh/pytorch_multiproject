import os
import logging
import numpy as np
from PIL import Image
from PIL import ImageDraw


class DetectionEvaluator:

    def __init__(self, save_dir):
        """
        To calculate the object detection scores, we accumulate ground truth labels (targets) and
           predictions of our model during the val phase, then compute the necessary metrics (e.g. bounding box mAP)

        Parameters
        ----------
        save_dir

        """
        self.logger = logging.getLogger(os.path.basename(__file__))
        self.data = []
        self.save_dir = save_dir

    def accumulate(self, save_img, targets, predictions):
        """

        Parameters
        ----------
        save_img (numpy.ndarray):
        targets (dict):
        predictions (dict):
        """
        self.data.append([save_img, targets, predictions])

    def bbox_score(self, iou_threshold=0.5, non_max_iou_thresh=0.5, score_threshold=0.6):
        """

        Parameters
        ----------
        iou_threshold
        non_max_iou_thresh
        score_threshold

        Returns
        -------

        """
        remaning_idx = []
        true_positive = np.array([])
        false_positive = np.array([])
        num_ground_truths = 0

        for _, targets, predictions in self.data:
            bboxes_targets = targets['boxes']
            bboxes_pred = predictions['boxes']
            bboxes_pred_score = predictions['scores']

            if not isinstance(bboxes_targets, np.ndarray):
                bboxes_targets = bboxes_targets.numpy()
            if not isinstance(bboxes_pred, np.ndarray):
                bboxes_pred = bboxes_pred.numpy()
            if not isinstance(bboxes_pred_score, np.ndarray):
                bboxes_pred_score = bboxes_pred_score.numpy()

            # apply non-max suppression to predictions
            bboxes_pred_suppr, _, idx = self.non_max_suppr_binary(bboxes_pred_score, bboxes_pred,
                                                                  score_threshold, non_max_iou_thresh)
            remaning_idx.append(idx)
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

                guessed = False
                for target_idx, iou in enumerate(iou_group):
                    if iou > iou_threshold and target_idx not in guessed_bboxes:
                        guessed_bboxes.append(target_idx)
                        true_pos_iter[group_idx] += 1
                        guessed = True

                    # if the prediction guessed no bboxes and we are at the end of the list
                    # count it as fp
                    if guessed is False and target_idx == (len(iou_group) - 1):
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

        avg_precision, _, _ = self.get_average_precision(precision, recall)

        self.logger.debug('\n\n')
        return avg_precision, precision[-1], recall[-1], remaning_idx

    def mask_score(self):
        """
            Calculates the mAP score for the generated masks.
        """
        raise NotImplementedError

    def non_max_suppr_binary(self, bboxes_pred_score, bboxes_pred, score_threshold, iou_threshold):
        """
        Binary classification version of non-max suppression

        Parameters
        ----------
        bboxes_pred_score
        bboxes_pred
        score_threshold
        iou_threshold

        Returns
        -------

        """

        remaining_idx = np.arange(bboxes_pred_score.shape[0])
        # firstly we discard all bbox predictions where class prob < base_treshold
        selected_idx = np.argwhere(bboxes_pred_score > score_threshold).flatten()
        selected_bboxes = bboxes_pred[selected_idx]
        selected_scores = bboxes_pred_score[selected_idx]
        remaining_idx = remaining_idx[selected_idx]

        out_bboxes = np.empty((0, 4))
        out_scores = np.array([])
        out_idx = np.array([])

        # continue iterations until the list of scores is depleted
        while len(selected_scores) > 0:
            highest_score_idx = np.argmax(selected_scores)

            top_score = selected_scores[highest_score_idx]
            top_bbox = selected_bboxes[highest_score_idx]
            top_idx = remaining_idx[highest_score_idx]

            selected_scores = np.delete(selected_scores, highest_score_idx)
            selected_bboxes = np.delete(selected_bboxes, highest_score_idx, axis=0)
            remaining_idx = np.delete(remaining_idx, highest_score_idx)

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
                remaining_idx = np.delete(remaining_idx, duplicate_boxes_idx)

            out_scores = np.append(out_scores, top_score)
            out_bboxes = np.append(out_bboxes, top_bbox.reshape(1, -1), axis=0)
            out_idx = np.append(out_idx, top_idx)

        return out_bboxes, out_scores, out_idx

    def save_bboxes_masks(self, epoch, selected_boxes_ind=None, mask_draw_precision=0.4, opacity=0.4):
        """
        Draws bounding boxes and masks on top of the original image and saves the result.
        Parameters
        ----------
        epoch (int): epoch number
        selected_boxes_ind (list): a list of lists containing indexes of selected bounding boxes
                    (after non-max suppression) for each image.
        mask_draw_precision (float, 0. to 1.): confidence score, above which the mask will be drawn
                    (each pixel in predicted mask is assigned a confidence score ranging from 0 to 1)
        opacity (float, 0. to 1.):  mask opacity, 0 - completely transparent, 1 - completely opaque
        """

        image_prep_list = list(self.draw_bbox(selected_boxes_ind))
        image_prep_list = list(self.generate_masked_img(image_prep_list,
                                                        selected_boxes_ind,
                                                        mask_draw_precision,
                                                        opacity))

        for idx, image in enumerate(image_prep_list):
            save_addr = os.path.join(self.save_dir, 'Test_img_{}_{}'.format(epoch, idx))
            image.save(save_addr, 'PNG')

    def draw_bbox(self, selected_boxes_ind=None):
        """
            Generator method.
            Draws bounding boxes on top of original image (green - ground truth, red - predicted bounding box).
            Yields resulting images
            Parameters
            ----------
            selected_boxes_ind (list).
        """
        for idx, data in enumerate(self.data):
            image, targets, predictions = data

            pred_bboxes = predictions['boxes']
            # select only bboxes remaining after suppression
            if selected_boxes_ind is not None:
                idx_group = selected_boxes_ind[idx]
                pred_bboxes = pred_bboxes[idx_group]

            targets_bboxes = targets['boxes']

            if not isinstance(targets_bboxes, np.ndarray):
                targets_bboxes = targets_bboxes.numpy()

            image_prep = Image.fromarray(image)
            draw = ImageDraw.Draw(image_prep)

            for target_bbox in targets_bboxes:
                draw.rectangle((tuple(target_bbox[:2]), tuple(target_bbox[2:])),
                               outline='green')

            for single_pred in pred_bboxes:
                draw.rectangle((tuple(single_pred[:2]), tuple(single_pred[2:])),
                               outline='red')
            yield image_prep

    def generate_masked_img(self, image_prep_list, selected_boxes_ind=None, mask_draw_precision=0.4, opacity=0.4):
        """
        Generator method.
        Overlays all the generated masks on top of the original image. Yields resulting images.
        Parameters
        ----------
        selected_boxes_ind (list), mask_draw_precision (float, 0. to 1.), opacity (float, 0. to 1.).
        """
        for idx, data in enumerate(self.data):
            image = image_prep_list[idx]
            masks = data[2]['masks'].mul(255).byte().numpy()

            if isinstance(image, np.ndarray):
                image_prep = Image.fromarray(image)
            elif isinstance(image, Image.Image):
                image_prep = image
            else:
                raise TypeError('The provided image type must be PIL image of numpy.ndarray')
            # add alpha channel to the original image
            image_prep.putalpha(255)

            if selected_boxes_ind is not None:
                idx_group = selected_boxes_ind[idx]
                if idx_group.dtype != int:
                    idx_group = idx_group.astype(int)
                # pick only those masks that correspond to the bounding boxes after non-max suppression
                masks = masks[idx_group]

            for mask in masks:
                colors = self.generate_color_scheme()
                # firstly generate 3 color channels and alpha channel
                mask = np.repeat(mask, 4, axis=0)
                # replace ones at each color channel with respective color if mask probability > mask_draw_precision
                # and zero out the values below mask_draw_precision
                for channel in range(len(colors)):
                    bool_mask_keep = mask[channel] >= int(255*mask_draw_precision)
                    bool_mask_erase = mask[channel] < int(255*mask_draw_precision)

                    mask[channel][bool_mask_keep] = colors[channel]
                    mask[channel][bool_mask_erase] = 0
                # fill alpha channel values using R channel as a reference
                mask[3, :, :][mask[0, :, :] > 0] = int(255*opacity)
                mask[3, :, :][mask[0, :, :] == 0] = 0

                # convert the mask into H, W, C format
                mask = np.transpose(mask, (1, 2, 0))
                # convert the prepared mask into PIL Image object
                mask_prep = Image.fromarray(mask)
                # combine the mask and the image
                image_prep = Image.alpha_composite(image_prep, mask_prep)
            yield image_prep

    @staticmethod
    def generate_color_scheme():
        return np.random.choice(range(255), size=3)

    @staticmethod
    def get_average_precision(precision, recall):
        """

        Parameters
        ----------
        precision
        recall

        Returns
        -------

        """
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
        """

        Parameters
        ----------
        bbox_1
        bbox_2

        Returns
        -------

        """
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
        """

        Parameters
        ----------
        bbox_array_1
        bbox_array_2

        Returns
        -------

        """
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
