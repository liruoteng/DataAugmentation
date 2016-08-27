#!/usr/bin/python

# run rendering rain streaks on background iamge

import augmentation as aug

my_worker = aug.AugmentClass()
my_worker.setup_middlebury()
my_worker.render()
