import numpy as np
import matplotlib.pyplot as plt
from tools import read_predicted_boxes, read_ground_truth_boxes


def calculate_iou(prediction_box, gt_box):
    """Calculate intersection over union of single predicted and ground truth box.

    Args:
        prediction_box (np.array of floats): location of predicted object as
            [xmin, ymin, xmax, ymax]
        gt_box (np.array of floats): location of ground truth object as
            [xmin, ymin, xmax, ymax]

        returns:
            float: value of the intersection of union for the two boxes.
    """
    # TASK 2A
    
    smallest_max = 0
    biggest_min = 0
    # Compute intersection
# Xdir

    if(prediction_box[2]>=gt_box[2]):
        smallest_max = gt_box[2]
    else:
        smallest_max = prediction_box[2]

    if(prediction_box[0]>=gt_box[0]):
        biggest_min = prediction_box[0]
    else:
        biggest_min = gt_box[0]

    intersectionXdir = smallest_max - biggest_min 
    if (intersectionXdir<0):
        intersectionXdir = 0
# Ydir

    if(prediction_box[3]>=gt_box[3]):
        smallest_max = gt_box[3]
    else:
        smallest_max = prediction_box[3]

    if(prediction_box[1]>=gt_box[1]):
        biggest_min = prediction_box[1]
    else:
        biggest_min = gt_box[1]

    intersectionYdir = smallest_max - biggest_min 
    if (intersectionYdir<0):
        intersectionYdir = 0

# calculate intersection
    intersection = intersectionXdir*intersectionYdir

    # Compute union
    gt_boxArea = (gt_box[2]-gt_box[0])*(gt_box[3]-gt_box[1])
    prediction_boxArea = (prediction_box[2]-prediction_box[0])*(prediction_box[3]-prediction_box[1])
    union = gt_boxArea + prediction_boxArea - intersection

    iou = intersection/union
    assert iou >= 0 and iou <= 1
    return iou
    '''
    leftX = max(prediction_box[0], gt_box[0])
    rightX = min(prediction_box[2], gt_box[2])
    topY = max(prediction_box[1], gt_box[1])
    bottomY = min(prediction_box[3], gt_box[3])

    intersectionArea = max(0, rightX - leftX) * max(0, bottomY - topY)
	# Compute union
    prediction_area = (prediction_box[2] - prediction_box[0]) * (prediction_box[3] - prediction_box[1])
    gt_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])

    iou = intersectionArea / float(prediction_area + gt_area - intersectionArea)
    assert iou >= 0 and iou <= 1
    return iou
    '''
def calculate_precision(num_tp, num_fp, num_fn):
    """ Calculates the precision for the given parameters.
        Returns 1 if num_tp + num_fp = 0

    Args:
        num_tp (float): number of true positives
        num_fp (float): number of false positives
        num_fn (float): number of false negatives
    Returns:
        float: value of precision
    """
    if ((num_tp + num_fp) == 0):
        precision = 1
    else:
        precision = num_tp/(num_tp + num_fp)
    return precision


def calculate_recall(num_tp, num_fp, num_fn):
    """ Calculates the recall for the given parameters.
        Returns 0 if num_tp + num_fn = 0
    Args:
        num_tp (float): number of true positives
        num_fp (float): number of false positives
        num_fn (float): number of false negatives
    Returns:
        float: value of recall
    """

    if ((num_tp + num_fn) == 0):
        recall = 0
    else:
        recall = num_tp/(num_tp + num_fn)
    return recall


def get_all_box_matches(prediction_boxes, gt_boxes, iou_threshold):
    """Finds all possible matches for the predicted boxes to the ground truth boxes.
        No bounding box can have more than one match.

        Remember: Matching of bounding boxes should be done with decreasing IoU order!

    Args:
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns the matched boxes (in corresponding order):
        predictiod bounding boxes
            shape: [number ofn_boxes: (np.array of floats): list of predicte box matches, 4].
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of box matches, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    """
    # Find all possible matches with a IoU >= iou threshold

    best_prediction_boxes = []
    gt_boxes = []

    for gt in gt_boxes:
        best_prediction = None
        highest_iou = 0
        for pb in prediction_boxes:
            iou = calculate_iou(pb, gt)
            if (iou>=highest_iou and iou >= iou_threshold):

                best_prediction = pb
                highest_iou = iou
        if best_prediction is not None:
            best_prediction_boxes.append(best_prediction)
            gt_boxes.append(gt)

    return np.array(best_prediction_boxes), np.array(gt_boxes)

def calculate_individual_image_result(prediction_boxes, gt_boxes, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes,
       calculates true positives, false positives and false negatives
       for a single image.
       NB: prediction_boxes and gt_boxes are not matched!

    Args:
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns:
        dict: containing true positives, false positives, true negatives, false negatives
            {"true_pos": int, "false_pos": int, false_neg": int}
    """
    matched_prediction_boxes, matched_gt_boxes = get_all_box_matches(prediction_boxes, gt_boxes, iou_threshold)
    tp = len(matched_prediction_boxes)
    fp = len(prediction_boxes)-tp
    fn = len(gt_boxes)-tp



    dict = {
        "true_pos": tp,
        "false_pos": fp, 
        "false_neg": fn
    }
    return dict

def calculate_precision_recall_all_images(
    all_prediction_boxes, all_gt_boxes, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes for all images,
       calculates recall and precision over all images
       for a single image.
       NB: all_prediction_boxes and all_gt_boxes are not matched!

    Args:
        all_prediction_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all predicted bounding boxes for the given image
            with shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        all_gt_boxes: (list of np.array of floats): each element in the list
             is a np.array containing all ground truth bounding boxes for the given image
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns:
        tuple: (precision, recall). Both float.


    """
    
    tp = 0
    fp = 0
    fn = 0


    for i in range(len(all_prediction_boxes)):

        dict = calculate_individual_image_result(all_prediction_boxes[i], all_gt_boxes[i], iou_threshold)
        
        tp += dict["true_pos"]
        fp += dict["false_pos"]
        fn +=dict["false_neg"]
    


    precision = calculate_precision(tp, fp, fn)
    recall = calculate_recall(tp, fp, fn)
    return (precision,recall)



def get_precision_recall_curve(
    all_prediction_boxes, all_gt_boxes, confidence_scores, iou_threshold
):
    """Given a set of prediction boxes and ground truth boxes for all images,
       calculates the recall-precision curve over all images.
       for a single image.

       NB: all_prediction_boxes and all_gt_boxes are not matched!

    Args:
        all_prediction_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all predicted bounding boxes for the given image
            with shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        all_gt_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all ground truth bounding boxes for the given image
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        scores: (list of np.array of floats): each element in the list
            is a np.array containting the confidence score for each of the
            predicted bounding box. Shape: [number of predicted boxes]

            E.g: score[0][1] is the confidence score for a predicted bounding box 1 in image 0.
    Returns:
        precisions, recalls: two np.ndarray with same shape.
    """
    # Instead of going over every possible confidence score threshold to compute the PR
    # curve, we will use an approximation
    confidence_thresholds = np.linspace(0, 1, 500)
    

    precisions = [] 
    recalls = []
    
    
    for current_threshold in confidence_thresholds:
        use_prediction_boxes = []
        for j in range(len(all_prediction_boxes)):
            temp_predictions = []
            for r in range(len(all_prediction_boxes[j])):
                if confidence_scores[j][r]>=current_threshold:

                    temp_predictions.append(all_prediction_boxes[j][r])
            use_prediction_boxes.append(np.array(temp_predictions))

        res = calculate_precision_recall_all_images(use_prediction_boxes, all_gt_boxes, iou_threshold)

        precisions.append(res[0])
        recalls.append(res[1])

    return np.array(precisions), np.array(recalls)


def plot_precision_recall_curve(precisions, recalls):
    """Plots the precision recall curve.
        Save the figure to precision_recall_curve.png:
        'plt.savefig("precision_recall_curve.png")'

    Args:
        precisions: (np.array of floats) length of N
        recalls: (np.array of floats) length of N
    Returns:
        None
    """
    plt.figure(figsize=(20, 20))
    plt.plot(recalls, precisions)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim([0.8, 1.0])
    plt.ylim([0.8, 1.0])
    plt.savefig("precision_recall_curve.png")


def calculate_mean_average_precision(precisions, recalls):
    """ Given a precision recall curve, calculates the mean average
        precision.

    Args:
        precisions: (np.array of floats) length of N
        recalls: (np.array of floats) length of N
    Returns:
        float: mean average precision
    """
    # Calculate the mean average precision given these recall levels.
    recall_levels = np.linspace(0, 1.0, 11)
    '''
    # YOUR CODE HERE
    average_precision = 0.0
    sum_precisions = 0.0
    for i in range(len(recall_levels)):
        max_precision = 0.0

        for j in range(recalls.shape[0]):
            if (recalls[j] >= recall_levels[i]) and (precisions[j]>max_precision):
                max_precision = precisions[j]
        sum_precisions += max_precision
    average_precision = sum_precisions / float(len(recall_levels))
    return average_precision
    '''
    precisions_max_sum = 0

    for lvl in range(len(recall_levels)):
    	precision_max = 0

    	for n in range(recalls.shape[0]):
    		if (precisions[n] > precision_max) and (recalls[n] >= recall_levels[lvl]):
    			precision_max = precisions[n]

    	precisions_max_sum += precision_max

    average_precision = precisions_max_sum / float(len(recall_levels))

    return average_precision

def mean_average_precision(ground_truth_boxes, predicted_boxes):
    """ Calculates the mean average precision over the given dataset
        with IoU threshold of 0.5

    Args:
        ground_truth_boxes: (dict)
        {
            "img_id1": (np.array of float). Shape [number of GT boxes, 4]
        }
        predicted_boxes: (dict)
        {
            "img_id1": {
                "boxes": (np.array of float). Shape: [number of pred boxes, 4],
                "scores": (np.array of float). Shape: [number of pred boxes]
            }
        }
    """
    # DO NOT EDIT THIS CODE
    all_gt_boxes = []
    all_prediction_boxes = []
    confidence_scores = []

    for image_id in ground_truth_boxes.keys():
        pred_boxes = predicted_boxes[image_id]["boxes"]
        scores = predicted_boxes[image_id]["scores"]

        all_gt_boxes.append(ground_truth_boxes[image_id])
        all_prediction_boxes.append(pred_boxes)
        confidence_scores.append(scores)

    precisions, recalls = get_precision_recall_curve(
        all_prediction_boxes, all_gt_boxes, confidence_scores, 0.5)
    plot_precision_recall_curve(precisions, recalls)
    mean_average_precision = calculate_mean_average_precision(precisions, recalls)
    print("Mean average precision: {:.4f}".format(mean_average_precision))


if __name__ == "__main__":
    ground_truth_boxes = read_ground_truth_boxes()
    predicted_boxes = read_predicted_boxes()
    mean_average_precision(ground_truth_boxes, predicted_boxes)
