import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import cv2 as cv
from scipy.ndimage import gaussian_filter, median_filter

class TimeSurface:
    def __init__(self, all_events, width, height, decay_const, time_step, update_type="static"):
        self.w = width
        self.h = height
        self.decay_const = decay_const
        self.surf = np.zeros((height+1, width+1))
        self.last_index = 0
        self.all_events = all_events
        self.time_step = time_step
        self.update_type = update_type

    def new_surf(self, time=0):
        surf = np.zeros((self.h+1, self.w+1))
        for i in range(int(time/self.time_step)):
            self.update_surf()
        return self.surf
    
    def update_surf(self):
        bool_array = self.all_events["time"].iloc[self.last_index:] < (self.all_events["time"].iloc[self.last_index] + self.time_step)
        events_to_add = self.all_events.iloc[self.last_index:].loc[bool_array]
        adj_tau = self.decay_const / len(events_to_add) if self.update_type == "proportional" else self.decay_const
        # removed code:  * (self.all_events["time"].iloc[self.last_index] - events_to_add["time"] + self.time_step)
        polarities = (events_to_add["polarity"] * 2 - 1) * np.exp(-1 * (self.all_events["time"].iloc[self.last_index] - events_to_add["time"] + self.time_step) / adj_tau / self.time_step)

        # NOTE: getting blockiness because events are all added, then scaled by the same amount. First multiply ndarray, then add events
        # scaled by their respective time intervals by adj_tau
        # yay i did it
        self.surf *= np.exp(-1 / adj_tau)
        
        np.put(self.surf, [(events_to_add["y"]) * (self.w+1) + events_to_add["x"]], polarities)
        
        self.surf.reshape((self.h+1, self.w+1))
        self.last_index = events_to_add.tail(1).index[0]

    def update_and_get_frame(self):
        self.update_surf()
        return self.surf

    def cv_display(self, time_to_display, frame_width, frame_height, speedup=1, blur_type="none"):
        if (blur_type[:6] == "median"):
            def processing(frame):
                return median_filter(frame, size=int(blur_type[7]))
        else:
            def processing(frame):
                return frame
        for i in range(int(time_to_display/self.time_step)):
            time_1 = time.time()
            frame = (self.update_and_get_frame() + 1)/2
            resized = cv.resize(processing(frame), (frame_width, frame_height))
            time_2 = time.time()
            while(time_2 - time_1 < self.time_step/speedup):
                time_2 = time.time()
            cv.imshow(blur_type, resized)
            # print(time_2 - time_1)
            if cv.waitKey(1) == ord('q'):
                break
        cv.destroyAllWindows()

    def cv_record(self, time_to_display, path, blur_type="none"):
        if (blur_type[:6] == "median"):
            def processing(frame):
                return median_filter(frame, size=int(blur_type[7]))
        else:
            def processing(frame):
                return frame
        num_digits = 8
        for i in range(int(time_to_display/self.time_step)):
            frame = (self.update_and_get_frame() + 1)/2
            resized = cv.resize(processing(frame), (self.w, self.h))
            cv.imwrite(path + "0"*(num_digits - len(str(i))) + str(i) + ".jpg", frame*255)
            

    # def cv_record((self, start_time, end_time, frame_width, frame_height, blur_type="none"))