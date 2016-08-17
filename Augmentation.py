#! /usr/python/bin

# data augmentation
import numpy as np
from PIL import Image
from PIL import ImageChops
import os


x = []
height = []
width = []
angle = []
#data_dir = '../../../../../media/liruoteng/File/KITTI/data_scene_flow/training/image_3'
#des_dir = '../../../../../media/liruoteng/File/KITTI/Rain/Training/image_3/'
data_dir = 'other-data'
des_dir = 'other-data-rain'
rain_name_list = ['001_R_3.png', '002_R_3.png', '003_R_3.png', '004_R_3.png', '005_R_3.png', '006_R_3.png',
                  '007_R_3.png', '008_R_3.png', '009_R_3.png', '010_R_3.png', '011_R_3.png', '012_R_3.png']
folder_list = os.listdir(data_dir)
file_list = []
for folder in folder_list:
    file_list.append(os.path.join(data_dir, folder, 'frame10.png'))
    file_list.append(os.path.join(data_dir, folder, 'frame11.png'))
# file_list.sort()
n = len(file_list)  # number of training images
rain_index = np.floor((np.random.rand(n)) * 12).astype(np.uint8)

for i in range(n):
    x.append(int(np.floor(np.random.normal(0, 20))))
    height.append(int(np.floor(np.random.normal(720, 40))))
    width.append(int(np.floor(np.random.normal(960, 60))))
    angle.append(np.random.normal(0, 1))


for i in range(n):
    rain_name = 'RainStreak/' + rain_name_list[rain_index[i]]
    gt_name = file_list[i]
    img = Image.open(rain_name, 'r')
    gt = Image.open(os.path.join(gt_name), 'r')
    img_offset = ImageChops.offset(img, x[i], 0)
    img_resize = img_offset.resize((width[i], height[i]), resample=Image.BILINEAR)
    (w, h) = img_resize.size
    (gw, gh) = gt.size
    left = w/2 - gw/2
    if gw % 2 == 1:
        right = w/2 + gw/2 + 1
    else:
        right = w/2 + gw/2
    top = h/2 - gh/2
    if gh % 2 == 1:
        bottom = h/2 + gh/2 + 1
    else:
        bottom = h/2 + gh/2
    img_rotate = img_resize.rotate(angle[i], expand=0)
    img_final = img_rotate.crop((left, top, right, bottom))
    rain_rgb = img_final.convert('RGB')
    rain_data = np.array(rain_rgb)
    gt_data = np.array(gt)
    input_data = gt_data.astype(np.uint16) + (rain_data - 128).astype(np.uint16)
    result = Image.fromarray(np.clip(input_data, 0, 255).astype(np.uint8))
    pos = gt_name.find('.')
    root_dir = os.path.dirname(gt_name)
    input_name = gt_name[:pos] + '_rain' + '.png'
    result.save(os.path.join(input_name))
    if i % 100 == 0:
        print "Processed", i

