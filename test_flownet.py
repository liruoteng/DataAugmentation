#! /usr/bin/python
import os, sys
from scripts.flownet import FlowNet
from utils import flowlib as fl


my_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(my_dir + '/..')

outfile = 'flownets-pred-0000000.flo'

# KITTI test files
# kitti_root = '../../../../../media/data/LRT_Flow/KITTI/data_scene_flow/training/'
# kitti_images_list = os.listdir(kitti_root + 'image_2').sort()
# kitti_gt_list = os.listdir(kitti_root + 'flow_noc').sort()

# Middlebury test files
middlebury_image = 'models/flownet/other-data'
middlebury_gt = 'models/flownet/other-gt-flow'
img1_name = 'frame10_rain.png'
img2_name = 'frame11_rain.png'
folders = os.listdir(middlebury_image)
accumulated = 0

result = open('result.txt', 'wb')
for folder in folders:
    img_files = []
    img_files.append = os.path.join('other-data', folder, img1_name)
    img_files.append = os.path.join('other-data', folder, img2_name)
    gt = os.path.join(middlebury_gt, folder, 'flow10.flo')
    FlowNet.run(my_dir, img_files, './model_simple')
    est_flow = fl.read_flow('models/flownet/' + outfile)
    gt_flow = fl.read_flow(gt)
    epe = fl.flowAngErr(est_flow[:, :, 0], est_flow[:, :, 1], gt_flow[:, :, 0], gt_flow[:, :, 1])
    result.write(folder + ': ' + epe + '\n')
    accumulated += epe

print accumulated
result.close()

