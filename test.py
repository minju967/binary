import os
import tableprint as tp
import pandas as pd


csv_path = "/media/minju/CCF89352F8933A22/project/csv_data/20view_dataset_test.csv"
num_views = 20
raw_data = pd.read_csv(csv_path, names=['img_path', 'class'])
filepaths = []
for idx in range(len(raw_data)):
        filepaths.append(raw_data.iloc[idx, 0])

filepaths = sorted(filepaths)
num_obj = int(len(filepaths) / num_views)
new_dataset = []
objs = []
for i in range(num_obj):
        new_dataset.append(filepaths[i * num_views:(i + 1) * num_views])
        objs.append([filepaths[i * num_views].split('/')[-2], filepaths[i * num_views].split('/')[-1][:-8]])

test_set = new_dataset
print(objs)
# width = [30, 10, 10, 40, 10]
# header = ['Name'.center(width[0]), 'Target'.center(width[1]), 'Predict'.center(width[2]),
#           'Predict Processing'.center(width[3]), 'Result'.center(width[4])]
# print(tp.header(header, width=width, style='round'))
#
# classes = ['A', 'B', 'C', 'D', 'E']
# A = '고속전극/연삭/연삭전극/방전'
# B = '고속가공/연삭 '
# C = '고속전극/연삭/방전'
# D = '연삭'
# E = '고속전극/연삭/방전/와이어'
#
#
# data = ['obj_name'.center(width[0]), 'obj_cls'.center(width[1]), 'pred'.center(width[2]), A.ljust(28),
#         'True'.center(width[4])]
# data1 = ['obj_name'.center(width[0]), 'obj_cls'.center(width[1]), 'pred'.center(width[2]), B.ljust(34),
#         'True'.center(width[4])]
# data2 = ['obj_name'.center(width[0]), 'obj_cls'.center(width[1]), 'pred'.center(width[2]), C.ljust(32),
#         'True'.center(width[4])]
# data3 = ['obj_name'.center(width[0]), 'obj_cls'.center(width[1]), 'pred'.center(width[2]), D.ljust(38),
#         'True'.center(width[4])]
# data4 = ['obj_name'.center(width[0]), 'obj_cls'.center(width[1]), 'pred'.center(width[2]), E.ljust(29),
#         'True'.center(width[4])]
#
# print(tp.row(data, width=width))
# print(tp.row(data1, width=width))
# print(tp.row(data2, width=width))
# print(tp.row(data3, width=width))
# print(tp.row(data4, width=width))
#
# print(tp.bottom(5, width=width))