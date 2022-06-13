from torch.autograd import Variable
from collections import Counter
import torch
import cv2
import numpy as np

# class N_Test(object):
#     def __init__(self, model, test_data, n_class, model_name, num):
#         self.test_loader = test_data
#         self.n_classes = n_class
#         self.Model = model
#         self.Model.eval()
#         self.cls = model_name
#         self.num_file = num
#
#     def test(self):
#         test_target = []
#         test_predict = []
#
#         pbar = tqdm(total=self.num_file, desc='Analyzing %s model' % self.cls)
#         for _, data in enumerate(self.test_loader):
#             N,V,C,H,W = data[1].size()
#             in_data = Variable(data[1]).view(-1,C,H,W).cuda()
#
#             result = self.Model(in_data)
#
#             # when pred == 0: negaitve data else pred == 1: positive data
#             pred = torch.max(result, 1)[1]
#             pred = pred.cpu().tolist()
#             test_predict.extend(pred)
#             target = data[0].tolist()
#             test_target.extend(target)
#             pbar.update(1)
#         pbar.close()
#         return test_predict, test_target

class Test(object):
    def __init__(self, model, input_data, n_class, model_name):
        self.input = input_data
        self.n_classes = n_class
        self.Model = model
        self.Model.eval()
        self.cls = model_name

    def test(self):
        C = 3
        H = 224
        W = 224
        in_data = Variable(self.input).view(-1,C,H,W).cuda()

        result = self.Model(in_data)

        # when pred == 0: negaitve data else pred == 1: positive data
        pred = torch.max(result, 1)[1]
        pred = pred.item()
        return pred

class CD_Test(object):
    def __init__(self, model, input_data, n_class, model_name):
        self.input = input_data
        self.n_classes = n_class
        self.Model = model
        # self.Model.eval()
        # print(self.Model)

    def get_output_layers(self, model):
        layer_names = model.getLayerNames()
        # output_layers = []
        # for i in model.getUnconnectedOutLayers():
        #     idx = i - 1
        #     output_layers.append(layer_names[idx])
        output_layers = [layer_names[i - 1] for i in model.getUnconnectedOutLayers()]
        return output_layers

    def run_predict(self, image_path):
        # read input image
        # print(image_path)
        image = cv2.imread(image_path)
        image = cv2.resize(image, dsize=(608, 608))
        Width = image.shape[1]
        Height = image.shape[0]
        scale = 0.0039215686
        # print(Width, Height)

        blob = cv2.dnn.blobFromImage(image, scale, (Width, Height), (0, 0, 0), True, crop=False)
        self.Model.setInput(blob)
        outs = self.Model.forward(self.get_output_layers(self.Model))

        # initialization
        class_ids = []
        confidences = []
        boxes = []
        center_boxes = []
        conf_threshold = 0.4
        nms_threshold = 0.4

        for out in outs:
            for detection in out:
                scores = detection[5:]
                # print(scores)
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])
                    center_boxes.append([center_x, center_y, w, h])
        # print(class_ids)
        # apply non-max suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
        # go through the detections remaining after nms and draw bounding box
        for i in indices:
            # i = i[0]
            box = boxes[i]
            center_box = center_boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]

        return class_ids

    def return_thresholded_predict(self, predict_label):
        # print(predict_label)
        counter = Counter(predict_label)
        # print(counter)
        if counter[2] > 1:
            return 1, 0
        else:
            return 0, 1

    def run_obj_predict(self):
        predict_label = []
        # self.input: obj 20view images path
        for idx, img_path in enumerate(self.input):
            # if idx == 0:
            #     print(img_path.split('/')[-1])
            result = self.run_predict(img_path)
            predict_label.extend(result)

        predict = self.return_thresholded_predict(predict_label)
        return predict
