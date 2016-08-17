#! /usr/python/bin

# replicate rain
from PIL import Image
import numpy as np

rain_list = ['001_R.png', '002_R.png', '003_R.png', '004_R.png', '005_R.png', '006_R.png', '007_R.png', '008_R.png', '009_R.png', '010_R.png', '011_R.png', '012_R.png']


for r in rain_list:
    rain = Image.open('RainStreak/' + r, 'r')
    rain_data = np.array(rain)
    rain_dup_data = np.tile(rain_data, 3)
    rain_duplicated = Image.fromarray(rain_dup_data)
    rain_name = r[:r.find('.')]
    rain_duplicated.save(rain_name + '_3.png')

