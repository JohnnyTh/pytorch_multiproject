import os
import sys
ROOT_DIR = os.path.dirname(os.path.abspath(os.path.dirname('__file__')))
sys.path.append(ROOT_DIR)

import logging
import numpy as np
# from mock import patch
# from mock import MagicMock
from utils.detection_evaluator import DetectionEvaluator


test_bboxes = {'box_1': [4, 9, 17, 13],
               'box_2': [10, 12, 18, 16],
               'box_3': [10, 20, 18, 24],
               'box_4': [4, 18, 11, 25],
               'box_5': [2, 5, 6, 10],
               'box_6': [12.5, 2.5, 17.5, 5],
               'box_7': [16, 4, 20, 7],
               'box_8': [12, 13, 19, 17]}

# precomputed ious
iou = {'1_2': 16/99,
       '1_3': 0,
       '2_8': 28/57,
       '4_3': 10/99,
       '5_1': 6/94,
       '6_7': 5/36}

# precomputed average precision
AP1 = (0.5-0) * 1
AP2 = (0.75-0.5)*0.3333
AP3 = (1-0.75)*0
AP = AP1 + AP2 + AP3

# targets, predictions
evaluator_data = [[{'boxes': np.array([test_bboxes['box_1'],
                                       test_bboxes['box_8']])},
                   {'boxes': np.array([test_bboxes['box_2'],
                                      test_bboxes['box_3'],
                                       test_bboxes['box_5']]), 'scores': np.array([0.99, 0.87, 0.99])}],

                  [{'boxes': np.array([test_bboxes['box_4']])},
                   {'boxes': np.array([test_bboxes['box_3'],
                                      test_bboxes['box_2'],
                                      test_bboxes['box_1']]), 'scores': np.array([0.99, 0.87, 0.2])}],

                  [{'boxes': np.array([test_bboxes['box_6']])},
                   {'boxes': np.array([test_bboxes['box_7']]), 'scores': np.array([0.99])}]
                  ]

batched_target = np.array([test_bboxes['box_1'], test_bboxes['box_4'], test_bboxes['box_6']])
batched_pred = np.array([test_bboxes['box_2'], test_bboxes['box_3'], test_bboxes['box_5'], test_bboxes['box_7']])
batched_iou = np.array([[16/99], [0], [6/94], [0], [0], [10/99], [0], [0], [0], [0], [0], [5/36]])

targets_predictions_comb = np.hstack([np.repeat(batched_target, batched_pred.shape[0], axis=0),
                                      np.tile(batched_pred, (batched_target.shape[0], 1))])


def test_iou():
    test_evaluator = DetectionEvaluator()
    assert test_evaluator.intersection_over_union(test_bboxes['box_1'], test_bboxes['box_2']) == iou['1_2']
    assert test_evaluator.intersection_over_union(test_bboxes['box_1'], test_bboxes['box_3']) == iou['1_3']
    assert test_evaluator.intersection_over_union(test_bboxes['box_4'], test_bboxes['box_3']) == iou['4_3']
    assert test_evaluator.intersection_over_union(test_bboxes['box_5'], test_bboxes['box_1']) == iou['5_1']
    assert test_evaluator.intersection_over_union(test_bboxes['box_6'], test_bboxes['box_7']) == iou['6_7']


def test_iou_batched():
    test_evaluator = DetectionEvaluator()
    assert test_evaluator.batch_iou(targets_predictions_comb[:, :4], targets_predictions_comb[:, 4:]).all() == batched_iou.all()


def test_box_score(caplog):
    test_evaluator = DetectionEvaluator()
    test_evaluator.data = evaluator_data
    avg_precision = test_evaluator.bbox_score(iou_threshold=0.12, score_threshold=0.6)
    assert round(avg_precision, 4) == round(AP, 4)
