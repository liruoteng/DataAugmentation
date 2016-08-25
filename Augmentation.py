#! /usr/python/bin
"""
Data Augmentation Program
"""
import os
import numpy as np
from PIL import Image
from PIL import ImageChops


class AugmentClass:
    """
    Formula: I = B + SR
    """

    def __init__(self):
        self.image_dir = ''
        self.i_dir = ''
        self.s_dir = ''
        self.r_dir = ''
        self.i_suffix = ''
        self.s_suffix = ''
        self.r_suffix = ''
        self.rain_name_list = []
        self.folder_list = []
        self.image_list = []
        self.image_height = None
        self.image_width = None
        self.height_range = []
        self.width_range = []
        self.angle_range = []
        self.offset_range = []
        self.rain_texture_dir = ''

    def setup_drive_dataset(self):
        self.rain_texture_dir = 'RainStreak'
        self.image_dir = '../../../../../media/liruoteng/ELLA/DataSet/frames_cleanpass'
        self.i_dir = '../../../../../media/liruoteng/ELLA/DataSet/frames_cleanpass'
        self.s_dir = '../../../../../media/liruoteng/ELLA/DataSet/frames_cleanpass'
        self.r_dir = '../../../../../media/liruoteng/ELLA/DataSet/frames_cleanpass'
        self.i_suffix = '_rain'
        self.s_suffix = '_s'
        self.r_suffix = '_r'
        self.rain_name_list = [os.path.join(self.rain_texture_dir, f) for f in os.listdir(self.rain_texture_dir)]
        # self.rain_name_list = ['001_R_3.png', '002_R_3.png', '003_R_3.png', '004_R_3.png', '005_R_3.png',
        # '006_R_3.png', '007_R_3.png', '008_R_3.png', '009_R_3.png', '010_R_3.png', '011_R_3.png', '012_R_3.png']
        self.image_height = 540
        self.image_width = 960
        for root, dirs, files in os.walk(self.image_dir):
            for filename in files:
                self.image_list.append(os.path.join(root, filename))
        self.image_list.sort()
        self.generate_augmentation_list()

    def generate_augmentation_list(self):
        """
        image_size : image size in HxW
        x: x offset list
        height: height variation list
        width : width variation list
        angle : angle variation list
        """
        n = len(self.image_list)
        x_offset_bound = self.image_width / 10
        height_bound = int(self.image_height / 10)
        width_bound = int(self.image_width / 10)
        angle_bound = 2
        for i in range(n):
            self.offset_range.append(int(np.floor(np.random.normal(0, x_offset_bound))))
            self.height_range.append(int(np.floor(np.random.normal(self.image_height * 2, height_bound))))
            self.width_range.append(int(np.floor(np.random.normal(self.image_width * 2, width_bound))))
            self.angle_range.append(np.random.normal(0, angle_bound))

    def render(self):
        """
        render different types/sizes rain on the background images and save the rain information
        :return:
        """
        n = len(self.image_list)
        rain_image_num = len(self.rain_name_list)
        rain_image_index = np.floor((np.random.rand(n)) * rain_image_num).astype(np.uint8)

        for i in range(n):
            # Read rain texture image and background image one by one
            rain_image_name = self.rain_name_list[rain_image_index[i]]
            bg_image_name = self.image_list[i]
            rain_image = Image.open(rain_image_name, 'r')
            bg_image = Image.open(bg_image_name, 'r')

            # augment rain texture image
            rain_rgb = self.augment_rain(i, rain_image)

            # convert final rain image object and back ground image object to array
            rain_data = np.array(rain_rgb)
            bg_data = np.array(bg_image)
            rain_s_value = (rain_data - 128)
            rain_r_value = ((rain_data - 128) == 0)
            image_i_data = bg_data.astype(np.uint16) + rain_s_value.astype(np.uint16)
            output_i = Image.fromarray(np.clip(image_i_data, 0, 255).astype(np.uint8))
            output_s = Image.fromarray(rain_s_value.astype(np.uint8))
            output_r = Image.fromarray(rain_r_value.astype(np.uint8))

            # file path manipulation
            bg_image_basename = os.path.basename(bg_image_name)
            bg_image_dirname = os.path.dirname(bg_image_name)
            bg_image_filename = bg_image_basename[:bg_image_basename.find('.')]
            image_i_name = os.path.join(bg_image_dirname, bg_image_filename + self.i_suffix + '.png')
            image_s_name = os.path.join(bg_image_dirname, bg_image_filename + self.s_suffix + '.png')
            image_r_name = os.path.join(bg_image_dirname, bg_image_filename + self.r_suffix + '.png')
            # save output files
            output_i.save(image_i_name)
            output_r.save(image_r_name)
            output_s.save(image_s_name)
            if i % 100 == 0:
                print "Processed", i
            if i == n:
                print "In Total ", i, " images have been processed"

    def augment_rain(self, i, rain_image):
        # Manipulate on rain image for augmentation
        img_offset = ImageChops.offset(rain_image, self.offset_range[i], 0)
        img_resize = img_offset.resize((self.width_range[i], self.height_range[i]), resample=Image.BILINEAR)
        (rain_image_width, rain_image_height) = img_resize.size
        left = rain_image_width / 2 - self.image_width / 2
        if self.image_width % 2 == 1:
            right = rain_image_width / 2 + self.image_width / 2 + 1
        else:
            right = rain_image_width / 2 + self.image_width / 2
        top = rain_image_height / 2 - self.image_height / 2
        if self.image_height % 2 == 1:
            bottom = rain_image_height / 2 + self.image_height / 2 + 1
        else:
            bottom = rain_image_height / 2 + self.image_height / 2
        img_rotate = img_resize.rotate(self.angle_range[i], expand=0)
        img_final = img_rotate.crop((left, top, right, bottom))
        rain_rgb = img_final.convert('RGB')

        return rain_rgb
