#! /usr/python/bin
"""
Data Augmentation Program
"""
import os
import numpy as np
from PIL import Image
from PIL import ImageChops
from scipy.ndimage import filters


class AugmentClass(object):
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
        self.rain_image_index = None
        self.blur_radius = [0.5, 1.0, 2.0, 4.0]
        self.bg_image_num = None

    def setup_drive_dataset(self):
        self.rain_texture_dir = 'RainStreak/wide'
        self.image_dir = '../../../../../media/liruoteng/ELLA/DataSet/frames_cleanpass'
        self.i_dir = '../../../../../media/liruoteng/ELLA/DataSet/frames_cleanpass'
        self.s_dir = '../../../../../media/liruoteng/ELLA/DataSet/frames_cleanpass'
        self.r_dir = '../../../../../media/liruoteng/ELLA/DataSet/frames_cleanpass'
        self.i_suffix = '_rain'
        self.s_suffix = '_s'
        self.r_suffix = '_r'
        self.rain_name_list = [os.path.join(self.rain_texture_dir, f) for f in os.listdir(self.rain_texture_dir)]
        self.image_height = 540
        self.image_width = 960
        for root, dirs, files in os.walk(self.image_dir):
            for filename in files:
                self.image_list.append(os.path.join(root, filename))
        self.image_list.sort()
        self.bg_image_num = len(self.image_list)
        rain_image_num = len(self.rain_name_list)
        self.rain_image_index = np.floor((np.random.rand(self.bg_image_num)) * rain_image_num).astype(np.uint8)
        self.prepare_augmentation_list()

    def setup_drive_finalpass(self):
        self.rain_texture_dir = 'RainStreak/wide'
        self.image_dir = '../../../../../media/liruoteng/File/Drive/frames_finalpass'
        self.i_dir = '../../../../../media/liruoteng/File/Drive/frames_finalpass'
        self.s_dir = '../../../../../media/liruoteng/File/Drive/frames_finalpass'
        self.r_dir = '../../../../../media/liruoteng/File/Drive/frames_finalpass'
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
        self.bg_image_num = len(self.image_list)
        rain_image_num = len(self.rain_name_list)
        self.rain_image_index = np.floor((np.random.rand(self.bg_image_num)) * rain_image_num).astype(np.uint8)
        self.prepare_augmentation_list()

    def setup_middlebury(self):
        self.rain_texture_dir = 'RainStreak/norm'
        self.image_dir = 'other-data'
        self.i_dir = 'other-data'
        self.s_dir = 'other-data'
        self.r_dir = 'other-data'
        self.i_suffix= '_rain'
        self.s_suffix = '_s'
        self.r_suffix = '_r'
        self.rain_name_list = [os.path.join(self.rain_texture_dir, f) for f in os.listdir(self.rain_texture_dir)]
        self.image_height = 480
        self.image_width = 640
        for root, dirs, files in os.walk(self.image_dir):
            for filename in files:
                self.image_list.append(os.path.join(root, filename))
        self.image_list.sort()
        self.bg_image_num = len(self.image_list)
        rain_image_num = len(self.rain_name_list)
        self.rain_image_index = np.floor((np.random.rand(self.bg_image_num)) * rain_image_num).astype(np.uint8)
        self.prepare_augmentation_list()

    def prepare_augmentation_list(self):
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

    def render_all(self):
        """
        render different types/sizes rain on the background images and save the rain information
        :return:
        """

        for i in range(self.bg_image_num):
            #for blur in blur_radius:

            # Read rain texture image and background image one by one
            rain_image_name = self.rain_name_list[self.rain_image_index[i]]
            bg_image_name = self.image_list[i]
            rain_image = Image.open(rain_image_name, 'r')
            bg_image = Image.open(bg_image_name, 'r')

            # augment rain texture image
            rain_rgb = self.augment_rain(i, rain_image, bg_image)

            # convert final rain image object and back ground image object to array
            rain_data = np.array(rain_rgb)
            bg_data = np.array(bg_image)
            # Rain image with Gaussian Blur
            rain_data_blur_1 = np.array(filters.gaussian_filter(rain_data, 0.5))
            rain_data_blur_2 = np.array(filters.gaussian_filter(rain_data, 1))
            rain_data_blur_3 = np.array(filters.gaussian_filter(rain_data, 2))
            rain_data_blur_4 = np.array(filters.gaussian_filter(rain_data, 4))

            # rain intensity map S
            rain_s_value = (rain_data - 128)
            rain_blur1_s_value = (rain_data_blur_1 - 128)
            rain_blur2_s_value = (rain_data_blur_2 - 128)
            rain_blur3_s_value = (rain_data_blur_3 - 128)
            rain_blur4_s_value = (rain_data_blur_4 - 128)

            # rain mask map R
            rain_r_value = ((rain_data - 128) == 0)
            rain_blur1_r_value = ((rain_data_blur_1 - 128) == 0)
            rain_blur2_r_value = ((rain_data_blur_2 - 128) == 0)
            rain_blur3_r_value = ((rain_data_blur_3 - 128) == 0)
            rain_blur4_r_value = ((rain_data_blur_4 - 128) == 0)

            # Apply rain data on the background
            image_input_data = np.clip(bg_data.astype(np.uint16)[:, :, 0:3] + rain_s_value.astype(np.uint16), 0, 255).astype(np.uint8)
            image_blur1_input_data = np.clip(bg_data.astype(np.uint16)[:, :, 0:3] + rain_blur1_s_value.astype(np.uint16), 0,255).astype(np.uint8)
            image_blur2_input_data = np.clip(bg_data.astype(np.uint16)[:, :, 0:3] + rain_blur2_s_value.astype(np.uint16), 0, 255).astype(np.uint8)
            image_blur3_input_data = np.clip(bg_data.astype(np.uint16)[:, :, 0:3] + rain_blur3_s_value.astype(np.uint16), 0, 255).astype(np.uint8)
            image_blur4_input_data = np.clip(bg_data.astype(np.uint16)[:, :, 0:3] + rain_blur4_s_value.astype(np.uint16), 0, 255).astype(np.uint8)

            # generate output files
            self.write_image(bg_image_name, image_input_data, self.i_suffix)
            self.write_image(bg_image_name, image_blur1_input_data, self.i_suffix + '_0.5')
            self.write_image(bg_image_name, image_blur2_input_data, self.i_suffix + '_1.0')
            self.write_image(bg_image_name, image_blur3_input_data, self.i_suffix + '_2.0')
            self.write_image(bg_image_name, image_blur4_input_data, self.i_suffix + '_4.0')
            self.write_image(bg_image_name, rain_s_value.astype(np.uint8), self.s_suffix)
            self.write_image(bg_image_name, rain_blur1_s_value.astype(np.uint8), self.s_suffix + '_0.5')
            self.write_image(bg_image_name, rain_blur2_s_value.astype(np.uint8), self.s_suffix + '_1.0')
            self.write_image(bg_image_name, rain_blur3_s_value.astype(np.uint8), self.s_suffix + '_2.0')
            self.write_image(bg_image_name, rain_blur4_s_value.astype(np.uint8), self.s_suffix + '_4.0')
            self.write_image(bg_image_name, rain_r_value.astype(np.uint8), self.r_suffix)
            self.write_image(bg_image_name, rain_blur1_r_value.astype(np.uint8), self.r_suffix + '_0.5')
            self.write_image(bg_image_name, rain_blur2_r_value.astype(np.uint8), self.r_suffix + '_1.0')
            self.write_image(bg_image_name, rain_blur3_r_value.astype(np.uint8), self.r_suffix + '_2.0')
            self.write_image(bg_image_name, rain_blur4_r_value.astype(np.uint8), self.r_suffix + '_4.0')

            if i % 100 == 0:
                print "Processed", i
            if i == self.bg_image_num-1:
                print "In Total ", i, " images have been processed"

    def render_image(self, bg_image_name, rain_texture_name):
        # Read rain texture and background image
        rain_texture = Image.open(rain_texture_name, 'r')
        bg_image = Image.open(bg_image_name, 'r')

    def get_image_array(self, bg_image, rain_texture):
        """
        get both background and rain texture data in array
        :param i: background image index
        :return: a 2-tuple with background and rain texture
        """
        # augment rain texture on background image
        rain_rgb = self.augment_rain(i, rain_texture, bg_image)

        # convert final rain image object and back ground image object to array
        rain_data = np.array(rain_rgb)
        bg_data = np.array(bg_image)
        return bg_data, rain_data

    def augment_rain(self, i, rain_image, bg_image):
        """
        augment rain texture on the target background image
        :param i: background image index
        :param rain_image: rain texture file
        :param bg_image: background image to be rendered
        :return: rain texture in rgb channels
        """
        # Manipulate on rain image for augmentation
        (width, height) = bg_image.size
        img_offset = ImageChops.offset(rain_image, self.offset_range[i], 0)
        img_resize = img_offset.resize((self.width_range[i], self.height_range[i]), resample=Image.BILINEAR)
        (rain_image_width, rain_image_height) = img_resize.size
        left = rain_image_width / 2 - width / 2
        if width % 2 == 1:
            right = rain_image_width / 2 + width / 2 + 1
        else:
            right = rain_image_width / 2 + width / 2
        top = rain_image_height / 2 - height / 2
        if height % 2 == 1:
            bottom = rain_image_height / 2 + height / 2 + 1
        else:
            bottom = rain_image_height / 2 + height / 2
        img_rotate = img_resize.rotate(self.angle_range[i], expand=0)
        img_final = img_rotate.crop((left, top, right, bottom))
        rain_rgb = img_final.convert('RGB')

        return rain_rgb

    @staticmethod
    def apply_rain_texture(bg_data, rain_intensity_data):
        """
        apply rain texture on the background image
        :param bg_data: background data in array
        :param rain_intensity_data: rain texture intensity data in array
        :return: rain rendered background image
        """
        bg_data_uint16 = bg_data.astype(np.uint16)[:, :, 0:3]
        rain_data_uint16 = rain_intensity_data.astype(np.uint16)
        image_input_data = np.clip(bg_data_uint16 + rain_data_uint16, 0, 255).astype(np.uint8)
        return image_input_data

    @staticmethod
    def blur_rain_texture(rain_data, radius):
        """
        blur rain texture with gaussian filter
        :param rain_data: rain texture image in array
        :param radius: radius for gaussian filter
        :return: blurred rain texture array, intensity map and indicator map
        """
        if radius != 0:
            rain_data_blur = np.array(filters.gaussian_filter(rain_data, radius))
        else:
            rain_data_blur = rain_data
        rain_intensity_data = (rain_data_blur - 128)   # S map
        rain_indicator_data = ((rain_data - 128) == 0) # R map
        return rain_data_blur, rain_indicator_data, rain_intensity_data

    @staticmethod
    def blur_bg_image(bg_data, rain_data, radius):
        """
        blur background image
        :param bg_data: raw background image data in array
        :param rain_data: rain texture image in array
        :param radius: gaussian blur radius
        :return: blurred background image
        """
        if radius != 0:
            rain_data_blur = np.array(filters.gaussian_filter(rain_data, radius))
        else:
            rain_data_blur = rain_data

    @staticmethod
    def write_image(image_name, image_data, suffix):
        # file path manipulation
        image_basename = os.path.basename(image_name)
        image_dirname = os.path.dirname(image_name)
        image_filename = image_basename[:image_basename.find('.')]
        image_out_name = os.path.join(image_dirname, image_filename + suffix + '.png')
        # save output files
        output = Image.fromarray(image_data)
        output.save(image_out_name)
        return