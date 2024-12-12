from modules import *

import numpy as np
import pandas as pd
import time
import cv2 as cv

from scipy.ndimage import gaussian_filter, median_filter

events = pd.read_csv("C:/Users/ernst/projects/imgs210-proj/events.txt", names=["time", "x", "y", "polarity"], delimiter=" ")
width = 240 # size of frame width in px
height = 180 # size of frame height in px
events["y"] = height - events["y"]
print(events.head(5), events.tail(5))


time_step_global = .1 # seconds
speed = 2
decay_const = 1

ts = TimeSurface(events, width, height, decay_const, time_step_global, update_type="proportional", threshold=0.1)

filtered_events = filter_by_density(ts, num_events=10000)
# filtered_events.to_csv("filtered_events.txt", index=False)
print(f'filtered events {filtered_events}')

ts2 = TimeSurface(filtered_events, width, height, decay_const, 10, update_type="static")
ts2.cv_display(59, 240*3, 180*3, speedup=speed, blur_type="medin 5", filter_type="defaul")