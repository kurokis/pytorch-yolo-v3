from __future__ import division
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
from util import *
from darknet import Darknet
from preprocess import prep_image, inp_to_image, letterbox_image
import pandas as pd
import random
import pickle as pkl
import argparse
from munkres import Munkres
import itertools


def get_test_input(input_dim, CUDA):
    img = cv2.imread("dog-cycle-car.png")
    img = cv2.resize(img, (input_dim, input_dim))
    img_ = img[:, :, ::-1].transpose((2, 0, 1))
    img_ = img_[np.newaxis, :, :, :]/255.0
    img_ = torch.from_numpy(img_).float()
    img_ = Variable(img_)

    if CUDA:
        img_ = img_.cuda()

    return img_


def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network.

    Returns a Variable
    """

    orig_im = img
    dim = orig_im.shape[1], orig_im.shape[0]
    img = (letterbox_image(orig_im, (inp_dim, inp_dim)))
    img_ = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim


def write(x, img):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    cls = int(x[-1])
    label = "{0}".format(object_classes[cls])
    color = random.choice(colors)
    cv2.rectangle(img, c1, c2, color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2, color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4),
                cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)
    return img


def arg_parse():
    """
    Parse arguements to the detect module

    """

    parser = argparse.ArgumentParser(
        description='YOLO v3 Video Detection Module')

    parser.add_argument("--video", dest='video', help="Video to run detection upon",
                        default="video.avi", type=str)
    parser.add_argument("--dataset", dest="dataset",
                        help="Dataset on which the network has been trained", default="pascal")
    parser.add_argument("--confidence", dest="confidence",
                        help="Object Confidence to filter predictions", default=0.5)
    parser.add_argument("--nms_thresh", dest="nms_thresh",
                        help="NMS Threshhold", default=0.4)
    parser.add_argument("--cfg", dest='cfgfile', help="Config file",
                        default="cfg/yolov3.cfg", type=str)
    parser.add_argument("--weights", dest='weightsfile', help="weightsfile",
                        default="yolov3.weights", type=str)
    parser.add_argument("--reso", dest='reso', help="Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default="416", type=str)
    return parser.parse_args()


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


class ImageObject:
    # Class variable
    latest_id = 0

    def __init__(self, bbox, class_):
        self.id = None
        self.bbox = bbox
        self.class_ = class_
        self.color = None

    def assign_id(self):
        self.id = ImageObject.latest_id
        ImageObject.latest_id += 1

        colors = pkl.load(open("pallete", "rb"))
        self.color = random.choice(colors)

    def calc_iou(self, another_object):
        """
        Returns intersection over union (IOU) among two bounding boxes

        Parameters
        ----------
        another_object: ImageObject

        Returns
        -------
        iou: float

        """

        boxA = self.bbox
        boxB = another_object.bbox

        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # compute the area of intersection rectangle
        interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
        if interArea == 0:
            return 0
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
        boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)

        # return the intersection over union value
        return iou


class Tracker:
    def __init__(self):
        self.max_objs = 16
        self.objs = []

    def track(self, prediction, img):
        # get bounding boxes from prediction
        tmp = prediction.cpu().detach().numpy().copy()
        bboxes = tmp[:, 1:5]
        classes = tmp[:, -1].astype(int)

        new_objs = [ImageObject(bbox, class_)
                    for bbox, class_ in zip(bboxes, classes)]

        # reject objects with high overlap for tracking robustness
        def remove_overlapping_objects(objs):
            objs_to_pop = []
            for i in range(len(objs)):
                for j in range(len(objs[i:])):
                    iou = new_objs[i].calc_iou(objs[i+j])
                    if 0.4 < iou < 1:
                        objs_to_pop.append(i+j)
            if len(objs_to_pop) > 0:
                # remove duplicates
                objs_to_pop = list(dict.fromkeys(objs_to_pop))
                # sort in reverse order
                objs_to_pop = sorted(objs_to_pop)[::-1]
                for x in objs_to_pop:
                    objs.pop(x)
        remove_overlapping_objects(new_objs)

        # return if no object in prediction
        if len(new_objs) == 0:
            return

        # if no object in tracker
        if len(self.objs) == 0:
            # store objects
            if len(new_objs) > self.max_objs:
                self.objs = new_objs[:self.max_objs]
            else:
                self.objs = new_objs
            # assign unique id to each object
            for obj in self.objs:
                obj.assign_id()
            return

        # assume at least one object for self.objs and new_objs from here on
        iou_matrix = np.array([[self.objs[i].calc_iou(new_objs[j]) for j in range(len(new_objs))]
                               for i in range(len(self.objs))])
        # append dummy zeros if new_objs < self.objs
        if len(new_objs) < len(self.objs):
            iou_matrix = np.concatenate([
                iou_matrix,
                np.zeros((len(self.objs), len(self.objs)-len(new_objs)))], axis=1)

        # Hungarian algorithm
        m = Munkres()
        result = m.compute(-1*iou_matrix.copy())

        assignment = np.array([[src, dst, iou_matrix[src][dst]]
                               for src, dst in result])

        # sort in descending order based on IOU
        assignment = assignment[np.argsort(assignment[:, 2])[::-1]]

        tracked_objects = []
        stray_objects = []
        for a in assignment:
            src = int(a[0])
            dst = int(a[1])
            iou = a[2]

            if iou > 0.2 and len(tracked_objects) < self.max_objs:
                # continue to track this object
                tracked_objects.append(self.objs[src])
                # update bbox location
                tracked_objects[-1].bbox = new_objs[dst].bbox
                # update class
                tracked_objects[-1].class_ = new_objs[dst].class_
            else:
                # consider the object as a new object
                # make sure dst does not point to a dummy object
                if dst < len(new_objs):
                    stray_objects.append(new_objs[dst])

        # append other objects which were kicked out from assignment
        stray_objects += [new_objs[i] for i in range(
            len(new_objs)) if i not in assignment[:, 1]]

        # pick up stray objects if there is room
        for s in stray_objects:
            if len(tracked_objects) < self.max_objs:
                # track the object
                tracked_objects.append(s)
                # assign a unique ID
                tracked_objects[-1].assign_id()

        remove_overlapping_objects(tracked_objects)

        self.objs = tracked_objects

        for i in range(len(self.objs)):
            obj = self.objs[i]
            c1 = tuple(obj.bbox[0:2].astype(int))
            c2 = tuple(obj.bbox[2:4].astype(int))
            cv2.rectangle(img, c1, c2, obj.color, 1)

            label = "{0}:{1}".format(str(obj.id), object_classes[obj.class_])
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
            c1 = c1[0], c1[1] - t_size[1] - 4
            c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
            cv2.rectangle(img, c1, c2, obj.color, -1)
            cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4),
                        cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)

        return


if __name__ == '__main__':
    tracker = Tracker()

    args = arg_parse()
    confidence = float(args.confidence)
    nms_thesh = float(args.nms_thresh)
    start = 0

    CUDA = torch.cuda.is_available()

    num_classes = 80

    CUDA = torch.cuda.is_available()

    bbox_attrs = 5 + num_classes

    print("Loading network.....")
    model = Darknet(args.cfgfile)
    model.load_weights(args.weightsfile)
    print("Network successfully loaded")

    model.net_info["height"] = args.reso
    inp_dim = int(model.net_info["height"])
    assert inp_dim % 32 == 0
    assert inp_dim > 32

    if CUDA:
        model.cuda()

    model(get_test_input(inp_dim, CUDA), CUDA)

    model.eval()

    videofile = args.video

    cap = cv2.VideoCapture(videofile)

    assert cap.isOpened(), 'Cannot capture source'

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    videowriter = cv2.VideoWriter(
        'output.avi', fourcc, fps, (W, H))

    counter = 0
    frames = 0
    start = time.time()
    while cap.isOpened():
        counter += 1
        ret, frame = cap.read()
        if ret:
            if(counter % 1 != 0):
                continue

            img, orig_im, dim = prep_image(frame, inp_dim)

            im_dim = torch.FloatTensor(dim).repeat(1, 2)

            if CUDA:
                im_dim = im_dim.cuda()
                img = img.cuda()

            with torch.no_grad():
                output = model(Variable(img), CUDA)
            output = write_results(
                output, confidence, num_classes, nms=True, nms_conf=nms_thesh)

            if type(output) == int:
                frames += 1
                print("FPS of the video is {:5.2f}".format(
                    frames / (time.time() - start)))
                cv2.imshow("frame", orig_im)
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break
                continue

            im_dim = im_dim.repeat(output.size(0), 1)
            scaling_factor = torch.min(inp_dim/im_dim, 1)[0].view(-1, 1)

            output[:, [1, 3]] -= (inp_dim - scaling_factor *
                                  im_dim[:, 0].view(-1, 1))/2
            output[:, [2, 4]] -= (inp_dim - scaling_factor *
                                  im_dim[:, 1].view(-1, 1))/2

            output[:, 1:5] /= scaling_factor

            for i in range(output.shape[0]):
                output[i, [1, 3]] = torch.clamp(
                    output[i, [1, 3]], 0.0, im_dim[i, 0])
                output[i, [2, 4]] = torch.clamp(
                    output[i, [2, 4]], 0.0, im_dim[i, 1])

            object_classes = load_classes('data/coco.names')
            colors = pkl.load(open("pallete", "rb"))

            tracker.track(output, orig_im)

            videowriter.write(orig_im)

            cv2.imshow("frame", orig_im)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            frames += 1
            print("FPS of the video is {:5.2f}".format(
                frames / (time.time() - start)))

        else:
            break
    videowriter.release()
