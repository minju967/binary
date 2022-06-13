import argparse
import os
import glob
import sys

import cv2
import tableprint as tp
import numpy as np
import torch
import csv
import pandas as pd

from tqdm import tqdm
from model.Models import SVCNN, MVCNN
from Tools.Trainer import Test, CD_Test
from Image import create_img
from PIL import Image
from torchvision import transforms
parser = argparse.ArgumentParser()

# create_image
parser.add_argument("-obj_path", "--path", type=str, default="/home/minju/project/UJU/OBJ")
parser.add_argument("-img_path", type=str, default='/home/minju/project/UJU/Image/')
parser.add_argument("-bs", "--batchSize", type=int, help="Batch size for the second stage", default=6)
parser.add_argument("-cnn_name", "--cnn_name", type=str, help="cnn model name", default="vgg")
parser.add_argument("-num_views", type=int, help="number of views", default=20)
parser.add_argument("-num_class", type=int, help="number of class", default=2)
parser.add_argument("-csv_path", type=str, default="/home/minju/project/UJU/_test.csv")
parser.add_argument("-num_test_class", type=int, help="number of class", default=5)
parser.add_argument("-val_path", type=str, default="/media/minju/CCF89352F8933A22/project/csv_data/20view_dataset_test.csv")      # 기존 이미지

# yolo
# parser.add_argument('-i', '--image_path', required=True, help ='path to input image')
parser.add_argument('-c', '--config', required=False, default='./yolo_evaluate_v2/yolov3-obj-5.cfg', help = 'path to yolo config file')
parser.add_argument('-w', '--weights', required=False, default='./yolo_evaluate_v2/yolov3-obj-5_best.weights', help = 'path to yolo pre-trained weights')
parser.add_argument('-cl', '--classes', required=False, help = 'path to text file containing class names')
parser.add_argument('-v', '--view', required=False, default=20, help = 'image view num per each obj')

# def main(args, image_list):
#     # args = parser.parse_args()
#     print('=' * 130)
#     print("Analzing the data ".center(130))
#     print('='*130)
#     val_dataset = MultiviewImgDataset(args.csv_path, args.num_views)
#     val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batchSize, shuffle=False, num_workers=0)
#
#     # print('num_test_files: ' + str(len(val_dataset.filepaths) // 6))
#     s_Model = SVCNN(args.num_class, args.cnn_name).cuda()
#
#     Model_A = MVCNN(s_Model, args.num_class, args.num_views, args.cnn_name).cuda()
#     Model_B = MVCNN(s_Model, args.num_class, args.num_views, args.cnn_name).cuda()
#     Model_C = MVCNN(s_Model, args.num_class, args.num_views, args.cnn_name).cuda()
#     Model_E = MVCNN(s_Model, args.num_class, args.num_views, args.cnn_name).cuda()
#
#     Model_A.load_state_dict(torch.load('./dict/model_A85.pt'))
#     Tester = N_Test(Model_A, test_data=val_loader, n_class=args.num_test_class, model_name='A', num=len(val_dataset.filepaths)//args.num_views//args.batchSize)
#     A_result, _ = Tester.test()
#     Model_B.load_state_dict(torch.load('./dict/model_B87.pt'))
#     Tester = N_Test(Model_B, test_data=val_loader, n_class=args.num_test_class, model_name='B', num=len(val_dataset.filepaths)//args.num_views//args.batchSize)
#     B_result, _ = Tester.test()
#     Model_C.load_state_dict(torch.load('./dict/model_C80.pt'))
#     Tester = N_Test(Model_C, test_data=val_loader, n_class=args.num_test_class, model_name='C', num=len(val_dataset.filepaths)//args.num_views//args.batchSize)
#     C_result, _ = Tester.test()
#     Model_E.load_state_dict(torch.load('./dict/model_E94.pt'))
#     D_result = torch.zeros(len(val_dataset.filepaths) // 6)
#     Tester = N_Test(Model_E, test_data=val_loader, n_class=args.num_test_class, model_name='E', num=len(val_dataset.filepaths)//args.num_views//args.batchSize)
#     E_result, target = Tester.test()
#
#     output = torch.stack([torch.tensor(A_result), torch.tensor(B_result), torch.tensor(C_result), D_result, torch.tensor(E_result)], dim=1)
#
#     names = []
#     for f in val_dataset.filepaths[::6]:
#         file = f.split('/')[-1]
#         file = file.split('_Y_')[0]+'.obj'
#         names.append(file)
#
#     predict = get_output(output=output)
#     target = torch.tensor(target)
#
#     visual_result(names, predict, target)

def create_input(inputs):
    tf = transforms.ToTensor()
    mean = [0.46147138, 0.45751584, 0.44702336]
    std = [0.20240466, 0.19746633, 0.18430763]

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=mean, std=std)])

    imgs = []
    for img in inputs:
        im = Image.open(img).convert('RGB')
        if transform:
            im = tf(im)
            im = transform(im)
        imgs.append(im)

    return torch.stack(imgs)

def main(args, img_list, obj):
    # args = parser.parse_args()
    input = create_input(img_list)  # image stack

    # print('num_test_files: ' + str(len(val_dataset.filepaths) // 6))
    s_Model = SVCNN(args.num_class, args.cnn_name).cuda()

    Model_A = MVCNN(s_Model, args.num_class, args.num_views, args.cnn_name).cuda()
    Model_B = MVCNN(s_Model, args.num_class, args.num_views, args.cnn_name).cuda()
    Model_E = MVCNN(s_Model, args.num_class, args.num_views, args.cnn_name).cuda()
    Yolo_CD = cv2.dnn.readNet(args.weights, args.config)

    # pbar = tqdm(total=4, desc="%s file Analyzing"%(obj))
    Model_A.load_state_dict(torch.load('./dict/model_A_20_view.pt'))
    Tester = Test(Model_A, input_data=input, n_class=args.num_test_class, model_name='A')
    A_result = Tester.test()
    # pbar.update(1)

    Model_B.load_state_dict(torch.load('./dict/model_B_20_view.pt'))
    Tester = Test(Model_B, input_data=input, n_class=args.num_test_class, model_name='B')
    B_result= Tester.test()
    # pbar.update(1)

    # Model_C.load_state_dict(torch.load('./dict/model_C_new1.pt'))
    # Tester = Test(Model_C, input_data=input, n_class=args.num_test_class, model_name='C')
    # C_result= Tester.test()
    # pbar.update(1)

    # C_result = 0
    # D_result = 0

    Model_E.load_state_dict(torch.load('./dict/model_E_20_view.pt'))
    Tester = Test(Model_E, input_data=input, n_class=args.num_test_class, model_name='E')
    E_result = Tester.test()
    # pbar.update(1)


    Tester = CD_Test(Yolo_CD, input_data=img_list, n_class=args.num_test_class, model_name='CD')
    C_result, D_result = Tester.run_obj_predict()  # 1 or 0 || 0 or 1
    # pbar.update(1)
    output = [A_result, B_result, C_result, D_result, E_result]     # [int, int, int, int, int]
    # pbar.close()
    return output

def get_output(output):
    result = []
    for i in range(len(output)):
        out = output[i]

        if out.sum() == 1:
            label = (out == 1).nonzero(as_tuple=True)[0].item()
        elif out.sum() > 1:
            out = out.tolist()
            idx = np.where(np.array(out)==1)[0]
            label = min(idx)
        else:
            label = 3
        result.append(label)

    result = torch.tensor(result, device='cpu')
    return result

def visual_result(name, predict, target):

    print('=' * 130)
    print("Processing result ".center(130))
    print('='*130)

    classes = ['A', 'B', 'C', 'D', 'E', 'F']
    process = ['고속전극 || 연삭 || 연삭전극 || 방전',
               '고속가공 || 연삭 ',
               '고속전극 || 연삭 || 방전',
               '연삭',
               '고속전극 || 연삭 || 방전 || 와이어',
               'ERROR']
    w = [28, 34, 32, 38, 29]
    file_names = name
    predict = predict
    width = [30, 10, 10, 40, 10]
    header = ['Name'.center(width[0]), 'Target'.center(width[1]), 'Predict'.center(width[2]), 'Predict Processing'.center(width[3]), 'Result'.center(width[4])]
    print(tp.header(header, width=width, style='round'))

    for idx in range(len(file_names)):
        obj_name = file_names[idx]
        obj_cls  = classes[target[idx]]
        pred = classes[predict[idx]]
        pros = str(process[predict[idx]])
        p_width = w[predict[idx]]
        result = obj_cls == pred
        data = [obj_name.center(width[0]), obj_cls.center(width[1]), pred.center(width[2]), pros.ljust(p_width), str(result).center(width[4])]
        print(tp.row(data, width=width))

    print(tp.bottom(5, width=width))
    print()

    results = predict == target
    wrong_class = np.zeros(5)
    samples_class = np.zeros(5)

    for i in range(results.size()[0]):
        if not bool(results[i].cpu().data.numpy()):
            wrong_class[target.cpu().data.numpy().astype('int')[i]] += 1
        samples_class[target.cpu().data.numpy().astype('int')[i]] += 1

    width = [10,40,20,20,10]
    w = [28, 34, 32, 38, 29]
    Header = ['Class'.center(width[0]), 'Processing'.center(width[1]), 'Number of data'.center(width[2]), 'Number of True data'.center(width[3]), 'Accuracy'.center(width[4])]
    print(tp.header(Header, width=width, style='round'))

    for idx in range(len(samples_class)):
        acc = str(round((samples_class[idx]-wrong_class[idx])/samples_class[idx]*100, 3))+'%'
        data = [classes[idx].center(width[0]), process[idx].ljust(w[idx]), str(int(samples_class[idx])).center(width[2]), str(int(samples_class[idx]-wrong_class[idx])).center(width[3]), acc.center(width[4])]
        print(tp.row(data, width=width))
    print(tp.bottom(5,width=width))

    return 0

# def create_view(args):
#     path = args.path
#     img_path = args.img_path
#     create_img(path,img_path)
#     return 0

def create_view(path):

    obj_file = path.split('/')[-1].replace('.obj','')
    img_path = args.img_path    # 이미지 저장 위치
    image_list = create_img(path,img_path)
    print('%s 이미지 생성완료 '%(path.split('/')[-1]))
    return image_list

def create_csv(args):
    f_test = open(args.csv_path, 'w', newline='')
    wr= csv.writer(f_test)

    data_path = args.img_path
    labels = os.listdir(data_path)
    labels = sorted(labels)

    for cls in labels:
        files = glob.glob(os.path.join(data_path,cls)+'/*.png')
        for i in range(len(files)):
            wr.writerow([files[i], cls])

def create_testset(csv_path, num_views):
    raw_data = pd.read_csv(csv_path, names=['img_path', 'class'])
    filepaths = []
    for idx in range(len(raw_data)):
        filepaths.append(raw_data.iloc[idx, 0])

    filepaths = sorted(filepaths)

    rand_idx = np.random.permutation(int(len(filepaths) / num_views))
    new_dataset = []
    objs = []

    for i in range(len(rand_idx)):
        new_dataset.append(filepaths[rand_idx[i] * num_views:(rand_idx[i] + 1) * num_views])
        objs.append([filepaths[rand_idx[i] * num_views].split('/')[-2], filepaths[rand_idx[i] * num_views].split('/')[-1][:-8]])
    # num_obj = int(len(filepaths)/num_views)
    # test_set = []
    # objs = []
    # for i in range(num_obj):
    #     test_set.append(filepaths[i * num_views:(i + 1) * num_views])
    #     objs.append([filepaths[i * num_views].split('/')[-2], filepaths[i * num_views].split('/')[-1][:-8]])

    return objs, new_dataset

if __name__ == '__main__':
    print()
    print()
    banner = 'Welcome to UJU Electronics'.center(130)
    tp.banner(banner, width=130)
    print()

    args = parser.parse_args()
    objs = []
    for (root, cls, files) in os.walk(args.path):
        for file in files:
            if '.obj' in file:
                objs.append(os.path.join(root,file))

    classes = ['A', 'B', 'C', 'D', 'E', 'F']
    process = ['고속전극 || 연삭 || 연삭전극 || 방전',
               '고속가공 || 연삭 ',
               '고속전극 || 연삭 || 방전',
               '연삭',
               '고속전극 || 연삭 || 방전 || 와이어',
               'ERROR']
    names = []
    target = []
    output = []

    objs, test_set = create_testset(args.val_path, args.num_views)
    correct = 0
    cnt = 0
    for obj, images in zip(objs, test_set):
        # obj: test_obj name
        # images: obj's 20view images list
        names.append(obj[1])
        cls = obj[0]
        if cls == 'c-train-20':
            cls = 'C'
        target.append(classes.index(cls))
        image_list = images
        predict = main(args, image_list, obj[1])
        if sum(predict) == 1:
            label = predict.index(1)
        elif sum(predict) > 1:
            # idx = np.where(np.array(predict)==1)[0]
            # idx = np.where(idx!=2 and idx !=3)
            predict[2], predict[3] = 0, 0
            # idx = np.array(predict)
            # idx = np.array(predict)[(idx<2)|(idx>3)]
            # print(idx)
            label = min(np.where(np.array(predict)==1)[0])
        else:
            label = 5

        predict = classes[label]
        processing = process[label]
        print("%s.obj File 예측 결과: %s || %s "%(obj[1], predict, obj[0]))
        print()
        output.append(label)

        if obj[0] == predict:
            correct += 1

        cnt += 1

        if label == 5:
            cnt -= 1

        if cnt % 5 == 0:
            print('Test accuracy: %.3f'%(correct/cnt))
            print()

    output = torch.tensor(output)
    target = torch.tensor(target)

    visual_result(names, output, target)

# if __name__ == '__main__':
def ex_main():
    print()
    print()
    banner = 'Welcome to UJU Electronics'.center(130)
    tp.banner(banner, width=130)
    print()

    args = parser.parse_args()
    objs = []
    for (root, cls, files) in os.walk(args.path):
        for file in files:
            if '.obj' in file:
                objs.append(os.path.join(root,file))

    classes = ['A', 'B', 'C', 'D', 'E']
    process = ['고속전극 || 연삭 || 연삭전극 || 방전',
               '고속가공 || 연삭',
               '고속전극 || 연삭 || 방전',
               '연삭',
               '고속전극 || 연삭 || 방전 || 와이어']
    names = []
    target = []
    output = []
    for obj in objs:
        names.append(obj.split('/')[-1])
        cls = obj.split('/')[-2]
        target.append(classes.index(cls))
        image_list = create_view(obj)
        predict = main(args, image_list, obj.split('/')[-1])
        if sum(predict) == 1:
            label = predict.index(1)
        elif sum(predict) > 1:
            idx = np.where(np.array(predict)==1)[0]
            idx = np.where(idx!=2 and idx !=3)
            label = min(idx)
        else:
            label = 2

        predict = classes[label]
        processing = process[label]
        print("%s File 예측 결과: %s [%s]"%(obj.split('/')[-1], predict, processing))
        print()
        output.append(label)

    output = torch.tensor(output)
    target = torch.tensor(target)

    visual_result(names, output, target)

