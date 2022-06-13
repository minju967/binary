import sys
import time

from PIL import Image
from tqdm import tqdm

import os
import pyvista as pv
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

def new_crop(file, mesh):
    img_rgb = cv2.imread(file, cv2.IMREAD_COLOR)
    min_x = 2000
    min_y = 2000
    max_x = 0
    max_y = 0

    for y in range(img_rgb.shape[0]):
        for x in range(img_rgb.shape[1]):
            if img_rgb[y][x][0] != 76:
                if x < min_x:
                    min_x = x
                if x > max_x:
                    max_x = x

                if y < min_y:
                    min_y = y
                if y > min_y:
                    max_y = y
            else:
                continue

    img = img_rgb[min_y:max_y, min_x:max_x]
    imgArray = np.array(img)

    cv2.imwrite(file, imgArray)
    creat_img(file)

def creat_img(file):
    img = Image.open(file)
    imgArray = np.array(img)

    m_len = max(list(imgArray.shape))

    base_y = np.full((20, imgArray.shape[1], 3),76)
    base_x = np.full((imgArray.shape[0], 20, 3), 76)


    if imgArray.shape[0] > imgArray.shape[1]:
        base = base_y
        new = np.append(base, imgArray, axis=0)
        new = np.append(new, base, axis=0)

        base = np.full((m_len+40, (new.shape[0]-new.shape[1])//2, 3),76)
        new = np.append(base, new, axis=1)
        new = np.append(new, base, axis=1)

    else:
        base = base_x
        new = np.append(base, imgArray, axis=1)
        new = np.append(new, base, axis=1)

        base = np.full(((new.shape[1]-new.shape[0])//2, m_len+40, 3), 76)
        new = np.append(base, new, axis=0)
        new = np.append(new, base, axis=0)

    ax = plt.gca()
    ax.axis('off')
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    plt.imshow((new * 1).astype(np.uint8))
    plt.savefig(file, bbox_inches="tight", pad_inches=0)
    plt.close()
    # print(file)

def rotation_Z(file, mesh, snap):
    p = pv.Plotter(off_screen=True)
    actor = p.add_mesh(mesh)
    p.store_image = True
    p.disable_parallel_projection()
    mesh.rotate_z(90)
    p.camera_position = (1.0, 0.0, 0.0)
    if snap == True:
        p.show(screenshot=file)
        # time.sleep(1)
    else:
        return

def rotation_Y(file, mesh, snap):
    p = pv.Plotter(off_screen=True)
    actor = p.add_mesh(mesh)
    p.store_image = True
    p.disable_parallel_projection()
    mesh.rotate_y(90)
    p.camera_position = (1.0, 0.0, 0.0)
    if snap == True:
        p.show(screenshot=file)
        # time.sleep(1)
    else:
        return

# def create_img(obj_path, save_path):
#     print('='*130)
#     print("Create Image".center(130))
#     print('='*130)
#     print()
#
#     path = obj_path
#     # cnt = 0
#     # for (_, _, files) in os.walk(path):
#     #     for file in files:
#     #         if '.obj' in file:
#     #             cnt += 1
#     # pbar = tqdm(total=cnt*6)
#
#     classes = os.listdir(path)
#     classes = sorted(classes)
#     for cls in classes:
#         # print('=' * 40)
#         # print("Create %c class Image".center(50)%cls)
#         # print('=' * 40)
#         # print()
#         save_img = os.path.join(save_path, cls)
#         if not os.path.exists(save_img):
#             os.makedirs(save_img, exist_ok=True)
#         meshes = os.listdir(os.path.join(path, cls))
#
#         pbar = tqdm(total=len(meshes) * 6, desc='Processing %s class create Image'%cls)
#
#         for idx in range(len(meshes)):
#             m = meshes[idx]
#             mesh = pv.read(os.path.join(path,cls,m))
#             p = pv.Plotter(off_screen=True)
#             actor = p.add_mesh(mesh)
#             p.camera_position = (1.0, 0.0, 0.0)
#             mesh.rotate_x(90)
#             p.show()
#
#             for i in range(4):
#                 snap = True
#                 f = save_img + '/' + m.replace('.obj', '_Z_') + str(i) + '.png'
#                 if os.path.exists(f):
#                     os.remove(f)
#
#                 rotation_Z(f, mesh, snap)
#                 new_crop(f, mesh)
#                 pbar.update(1)
#
#             for i in range(4):
#                 snap = True
#                 if i == 1 or i == 3:
#                     snap = False
#                 f = save_img + '/' + m.replace('.obj', '_Y_') + str(i) + '.png'
#                 if os.path.exists(f):
#                     os.remove(f)
#
#                 rotation_Y(f, mesh, snap)
#                 if snap == True:
#                     new_crop(f, mesh)
#                     pbar.update(1)
#         pbar.close()
#         print()

def create_img(obj_path, save_path):

    path = obj_path
    cls = path.split('/')[-2]
    obj_file = path.split('/')[-1]
    save_img = os.path.join(save_path, cls)

    if not os.path.exists(save_img):
        os.makedirs(save_img, exist_ok=True)

    pbar = tqdm(total=6, desc='Processing %s OBJ File 이미지 생성'%obj_file)

    mesh = pv.read(path)
    p = pv.Plotter(off_screen=True)
    actor = p.add_mesh(mesh)
    p.camera_position = (1.0, 0.0, 0.0)
    mesh.rotate_x(90)
    p.show()
    imgs = []
    for i in range(4):
        snap = True
        f = save_img + '/' + obj_file.replace('.obj', '_Z_') + str(i) + '.png'
        if os.path.exists(f):
            os.remove(f)

        rotation_Z(f, mesh, snap)
        new_crop(f, mesh)
        pbar.update(1)
        imgs.append(f)
    for i in range(4):
        snap = True
        if i == 1 or i == 3:
            snap = False
        f = save_img + '/' + obj_file.replace('.obj', '_Y_') + str(i) + '.png'
        if os.path.exists(f):
            os.remove(f)

        rotation_Y(f, mesh, snap)
        if snap == True:
            new_crop(f, mesh)
            pbar.update(1)
            imgs.append(f)
    pbar.close()

    fig1 = plt.figure()  # rows*cols 행렬의 i번째 subplot 생성
    rows = 2
    cols = 3
    i = 1

    for filename in imgs:
        fig1.suptitle('%s Image'%obj_file, fontsize=16)

        img = cv2.imread(filename)
        ax = fig1.add_subplot(rows, cols, i)
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_xticks([]), ax.set_yticks([])
        i += 1

    fig1.tight_layout()
    plt.show(block=False)
    plt.pause(3)
    plt.close()
    return imgs
