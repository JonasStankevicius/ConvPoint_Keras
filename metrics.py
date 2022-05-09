import numpy as np
import tensorflow as tf

#==============================================
#==============================================
# STATS
#==============================================
#==============================================


def stats_overall_accuracy(cm):
    """Compute the overall accuracy.
    """
    return np.trace(cm)/cm.sum()


def stats_pfa_per_class(cm):
    """Compute the probability of false alarms.
    """
    sums = np.sum(cm, axis=0)
    mask = (sums>0)
    sums[sums==0] = 1
    pfa_per_class = (cm.sum(axis=0)-np.diag(cm)) / sums
    pfa_per_class[np.logical_not(mask)] = -1
    average_pfa = pfa_per_class[mask].mean()
    return average_pfa, pfa_per_class


def stats_accuracy_per_class(cm):
    """Compute the accuracy per class and average
        puts -1 for invalid values (division per 0)
        returns average accuracy, accuracy per class
    """
    # equvalent to for class i to
    # number or true positive of class i (data[target==i]==i).sum()/ number of elements of i (target==i).sum()
    sums = np.sum(cm, axis=1)
    mask = (sums>0)
    sums[sums==0] = 1
    accuracy_per_class = np.diag(cm) / sums #sum over lines
    accuracy_per_class[np.logical_not(mask)] = -1
    average_accuracy = accuracy_per_class[mask].mean()
    return average_accuracy, accuracy_per_class


def stats_iou_per_class(cm, ignore_missing_classes=True):
    """Compute the iou per class and average iou
        Puts -1 for invalid values
        returns average iou, iou per class
    """

    sums = (np.sum(cm, axis=1) + np.sum(cm, axis=0) - np.diag(cm))
    mask  = (sums>0)
    sums[sums==0] = 1
    iou_per_class = np.diag(cm) / sums
    iou_per_class[np.logical_not(mask)] = -1

    if mask.sum()>0:
        average_iou = iou_per_class[mask].mean()
    else:
        average_iou = 0

    return average_iou, iou_per_class


def stats_f1score_per_class(cm):
    """Compute f1 scores per class and mean f1.
        puts -1 for invalid classes
        returns average f1 score, f1 score per class
    """
    # defined as 2 * recall * prec / recall + prec
    sums = (np.sum(cm, axis=1) + np.sum(cm, axis=0))
    mask  = (sums>0)
    sums[sums==0] = 1
    f1score_per_class = 2 * np.diag(cm) / sums
    f1score_per_class[np.logical_not(mask)] = -1
    average_f1_score =  f1score_per_class[mask].mean()
    return average_f1_score, f1score_per_class

class MIOU(tf.keras.metrics.Metric):
    
    def __init__(self, classCount, name='miou', **kwargs):
        super(MIOU, self).__init__(name=name, **kwargs)
        self.cm = self.add_weight(name=name, shape = (classCount, classCount), initializer='zeros', dtype = tf.int64)
        self.classCount = classCount

    def update_state(self, y_true, y_pred, sample_weight=None):
        TrueLbl = tf.argmax(tf.reshape(y_true, [-1, self.classCount]), axis= 1)
        PredLbl = tf.argmax(tf.reshape(y_pred, [-1, self.classCount]), axis= 1)
        confusion_matrix = tf.math.confusion_matrix(TrueLbl, PredLbl, self.classCount)          
        self.cm.assign_add(tf.cast(confusion_matrix, tf.int64))

    def result(self):
        union = tf.linalg.diag_part(self.cm)
        rowSum = tf.math.reduce_sum(self.cm, axis = 0)
        colSum = tf.math.reduce_sum(self.cm, axis = 1)
        intersection = (colSum + rowSum - union)
        intersection = tf.where(tf.equal(intersection, tf.constant(0, dtype=tf.int64)), tf.constant(1, dtype=tf.int64), intersection)
        iou =  union / intersection
        miou = tf.expand_dims(tf.convert_to_tensor(tf.reduce_sum(iou) / tf.cast(iou.shape[0], tf.float64)), 0)
        # return tf.cast(miou, tf.float64)
        return tf.concat((tf.expand_dims(miou,1), tf.cast(tf.expand_dims(iou,1), tf.float64)), 0)

    def reset_state(self):
      # The state of the metric will be reset at the start of each epoch.
      self.cm.assign(tf.zeros((self.classCount, self.classCount), dtype=tf.int64))

def moving_miou_metric(classCount):
    def moving_iou(y_true, y_pred):
        TrueLbl = tf.argmax(tf.reshape(y_true, [-1, classCount]), axis= 1)
        PredLbl = tf.argmax(tf.reshape(y_pred, [-1, classCount]), axis= 1)

        cm = tf.math.confusion_matrix(TrueLbl, PredLbl, classCount)

        union = tf.linalg.diag_part(cm)

        rowSum = tf.math.reduce_sum(cm, axis = 0)
        colSum = tf.math.reduce_sum(cm, axis = 1)

        intersection = (colSum + rowSum - union)+1

        iou =  union / intersection

        return tf.reduce_sum(iou) / tf.cast(tf.math.maximum(iou.shape[0], 1), dtype=np.float64)

    return moving_iou

class IOU(tf.keras.metrics.Metric):
    def __init__(self, classCount, classIndex, name='iou', **kwargs):
        super(IOU, self).__init__(name=name, **kwargs)
        self.cm = self.add_weight(name=name, shape = (classCount, classCount), initializer='zeros', dtype = tf.int64)
        self.classCount = classCount
        self.classIndex = classIndex

    def update_state(self, y_true, y_pred, sample_weight=None):
        TrueLbl = tf.argmax(tf.reshape(y_true, [-1, self.classCount]), axis= 1)
        PredLbl = tf.argmax(tf.reshape(y_pred, [-1, self.classCount]), axis= 1)
        confusion_matrix = tf.math.confusion_matrix(TrueLbl, PredLbl, self.classCount)
        self.cm.assign_add(tf.cast(confusion_matrix, tf.int64))

    def result(self):
        union = tf.linalg.diag_part(self.cm)
        rowSum = tf.math.reduce_sum(self.cm, axis = 0)
        colSum = tf.math.reduce_sum(self.cm, axis = 1)
        intersection = (colSum + rowSum - union)
        intersection = tf.where(tf.equal(intersection, tf.constant(0, dtype=tf.int64)), tf.constant(1, dtype=tf.int64), intersection)
        iou =  union / intersection
        return tf.cast(tf.expand_dims(iou, 1)[self.classIndex], tf.float64)

    def reset_state(self):
        # The state of the metric will be reset at the start of each epoch.
        self.cm.assign(tf.zeros((self.classCount, self.classCount), dtype=tf.int64))

def weighted_categorical_crossentropy(weights):
    # weights = [0.9,0.05,0.04,0.01]
    def wcce(y_true, y_pred):
        Kweights = tf.constant(weights)
        y_true = tf.cast(y_true, y_pred.dtype)
        return tf.keras.losses.categorical_crossentropy(y_true, y_pred) * tf.math.reduce_sum(y_true * Kweights, axis=-1)

    return wcce