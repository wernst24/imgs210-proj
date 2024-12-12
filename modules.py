import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import cv2 as cv
from scipy.ndimage import gaussian_filter, median_filter
from collections.abc import Callable

class TimeSurface:
    def __init__(self, all_events, width, height, decay_const, time_step, update_type="static", threshold=0.6, simga=1):
        self.w = width
        self.h = height
        self.decay_const = decay_const
        self.surf = np.zeros((height+1, width+1))
        self.filtered_surf = np.zeros((height+1, width+1))
        self.last_index = 0
        self.all_events = all_events
        self.time_step = time_step
        self.update_type = update_type
        self.threshold = threshold
        self.sigma = simga

    def new_surf(self, time=0):
        surf = np.zeros((self.h+1, self.w+1), dtype=np.float32)
        for i in range(int(time/self.time_step)):
            self.update_surf()
        return self.surf
    
    def filter_by_density(self, events_to_add):
        # print(events_to_add, type(events_to_add))
        blurred_surf = gaussian_filter(self.surf, sigma=self.sigma)
        filtered_events = events_to_add[abs(blurred_surf[events_to_add.y][events_to_add.x]) > self.threshold]
        # print(filtered_events.head(5), events_to_add.head(5))
        return filtered_events
    
    def put_filtered_events(self, events_to_add, polarities):
        blurred_surf = gaussian_filter(self.surf, sigma=self.sigma)
        
        for index, row in events_to_add.iterrows():
            x = row["x"]
            y = row["y"]
            if abs(blurred_surf[y][x] > self.threshold):
                np.put(self.surf, [y * (self.w+1) + x], polarities[index])

    def update_surf(self):
        bool_array = self.all_events["time"].iloc[self.last_index:] < (self.all_events["time"].iloc[self.last_index] + self.time_step)
        events_to_add = self.all_events.iloc[self.last_index:].loc[bool_array]
        # filtered_events_to_add = self.filter_by_density(events_to_add)
        # filtered_events_to_add = events_to_add


        adj_tau = self.decay_const / len(events_to_add) if self.update_type == "proportional" else self.decay_const
        # removed code:  * (self.all_events["time"].iloc[self.last_index] - events_to_add["time"] + self.time_step)
        last_time = self.all_events["time"].iloc[self.last_index]

        polarities = (events_to_add["polarity"] * 2 - 1) * np.exp(-1 * (last_time - events_to_add["time"] + self.time_step) / adj_tau / self.time_step)
        # filtered_polarities = (filtered_events_to_add["polarity"] * 2 - 1) * np.exp(-1 * (self.all_events["time"].iloc[self.last_index] - filtered_events_to_add["time"] + self.time_step) / adj_tau / self.time_step)

        # NOTE: getting blockiness because events are all added, then scaled by the same amount. First multiply ndarray, then add events
        # scaled by their respective time intervals by adj_tau
        # yay i did it
        self.surf *= np.exp(-1 / adj_tau)
        # self.filtered_surf *= np.exp(-1 / adj_tau)
        
        np.put(self.surf, [(events_to_add["y"]) * (self.w+1) + events_to_add["x"]], polarities)
        # self.put_filtered_events(events_to_add, polarities)
        # np.put(self.filtered_surf, [(filtered_events_to_add["y"]) * (self.w+1) + filtered_events_to_add["x"]], filtered_polarities)
        # self.filtered_surf[filtered_events_to_add["y"].to_numpy(), filtered_events_to_add["x"].to_numpy()] = filtered_polarities
        
        self.surf.reshape((self.h+1, self.w+1))
        # self.filtered_surf.reshape((self.h+1, self.w+1))
        self.last_index = events_to_add.tail(1).index[0]
    

    def update_and_get_frame(self, mode="default"):
        self.update_surf()
        return self.surf if mode == "default" else self.filtered_surf

    def cv_display(self, time_to_display, frame_width, frame_height, speedup=1, blur_type="none", filter_type="default"):
        if (blur_type[:6] == "median"):
            def processing(frame):
                return median_filter(frame, size=int(blur_type[7]))
        else:
            def processing(frame):
                return frame
        for i in range(int(time_to_display/self.time_step)):
            time_1 = time.time()
            frame = (self.update_and_get_frame(mode=filter_type) + 1)/2
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

# displays a grid of images with different functions applied to them
def imagegrid(images : list[np.array], funcs : Callable[[np.array], np.array], buffer_size=5):
    height_disp = images[0].shape[0]
    width_disp = images[0].shape[1]
    vertical_buffer = np.zeros((height_disp, buffer_size), dtype=np.float32) # vertical black lines
    horizontal_buffer = np.zeros((width_disp + buffer_size, buffer_size), dtype=np.float32).transpose()

    image_grid = [[np.concatenate([np.concatenate([func(np.array(image, dtype=np.float32)), vertical_buffer], axis=1), horizontal_buffer], axis=0) for func in funcs] for image in images]

    # images_f32 = [np.concatenate([np.concatenate([images[i], vertical_buffer], axis=1), horizontal_buffer], axis=0) for i in range(len(images))]
    image_rows = [np.concatenate(image_grid[i], axis = 1) for i in range(len(image_grid))]
    all_images = np.concatenate(image_rows, axis=0)
    try:
        while True:
            cv.imshow('all images', all_images)
            if cv.waitKey(1) == ord('q'):
                break
    finally:
        cv.destroyAllWindows()

import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter

def filter_events_by_density(events: pd.DataFrame, width: int, height: int, 
                             decay_const: float, time_step: float, 
                             sigma: float, threshold: float) -> pd.DataFrame:
    """
    Filters events based on the time surface density calculation.

    Parameters:
        events (pd.DataFrame): Dataframe of events with columns ['time', 'x', 'y', 'polarity'].
        width (int): Width of the time surface.
        height (int): Height of the time surface.
        decay_const (float): Decay constant for the exponential decay.
        time_step (float): Time step for updating the surface.
        sigma (float): Gaussian filter sigma for density estimation.
        threshold (float): Threshold for filtering based on density.

    Returns:
        pd.DataFrame: Filtered dataframe of events.
    """
    # Initialize the time surface
    time_surface = np.zeros((height + 1, width + 1), dtype=np.float32)

    # Initialize the filtered events list
    filtered_events = []

    # Iterate over the unique time windows in the event data
    start_time = events['time'].iloc[0]
    end_time = events['time'].iloc[-1]

    current_time = start_time
    last_index = 0

    while current_time <= end_time:
        # Select events in the current time window
        time_window = events[(events['time'] >= current_time) & (events['time'] < current_time + time_step)]
        
        if not time_window.empty:
            # Update the time surface with the selected events
            for _, event in time_window.iterrows():
                x, y, polarity = int(event['x']), int(event['y']), event['polarity']
                time_surface[y, x] += (polarity * 2 - 1)  # Convert polarity to +1/-1

            # Apply Gaussian blur for density estimation
            blurred_surface = gaussian_filter(time_surface, sigma=sigma)

            # Filter events based on density threshold
            for _, event in time_window.iterrows():
                x, y = int(event['x']), int(event['y'])
                if abs(blurred_surface[y, x]) > threshold:
                    filtered_events.append(event)

        # Move to the next time window
        current_time += time_step
        
        decay_factor = np.exp(-1 / decay_const)
        time_surface *= decay_factor

    # Convert the filtered events list back to a DataFrame
    return pd.DataFrame(filtered_events)

# Example usage:
# filtered_df = filter_events_by_density(events, width=240, height=180, decay_const=0.6, time_step=0.01, sigma=1, threshold=0.5)

def filter_by_density(ts : TimeSurface, num_events=10000):
    events = ts.all_events[:num_events]
    final_time = events["time"].iloc[-1]
    time_step = ts.time_step

    threshold = ts.threshold
    sigma = ts.sigma

    current_time = 0

    filtered_events = []

    while current_time <= final_time:
        # print(current_time)
        time_surface = gaussian_filter(ts.update_and_get_frame(), sigma=sigma)
        # print(time_surface.max())
        time_window = events[(events["time"] > current_time) & (events["time"] <= current_time + time_step)]
        for _, event in time_window.iterrows():
            x = int(event["x"])
            y = int(event["y"])
            if abs(time_surface[y, x]) > threshold:
                filtered_events.append(event)

        current_time += time_step
    return pd.DataFrame(filtered_events).reset_index(drop=True, inplace=True)