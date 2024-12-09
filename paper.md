# Background

## Event Cameras

Event cameras capure visual data in a fundamentally different way than traditional cameras. While traditional cameras capture whole-frame images at a fixed rate, such as 30 or 60 Hz, event cameras capture changes in pixel brightness asynchronously.

While this makes raw data difficult to visualize, event cameras have many advantages in a wide range of CV applications, due to their high dynamic range, low power consumpion, high temporal resolution, and low bandwidth requirements.

Due to the only recent availability of affordable and precise event cameras, there has been less time spent developing denoising algorithms for event data. Additionally, there are other unique constraints present when designing effective algorithms, namely the discrete nature of events.

### Methods of Representing Events

In order to better visualize event stream data, there have been various techniques developed to represent event streams as images. One common technique is the Time Surface (TS). This is an array which, at each point, stores the timestamp *t* of the last event at that position. When reading the time surface with an exponential kernel

`Image(x, y) = exp(-b * TimeSurface(x, y))`

more recent events are emphasized, and edges moving across the scene will leave trails. The decay constant, b, controls the length of the trails, and can be chosen as a function of mean events/second, or chosen experimentally.

In order to increase visibility, I will use a time surface with a decay constant proportional to the number of events per second, in order to reduce the trail size, and make images more consistent across various translation speeds.

Another modification I use in my implementation of the time surface is allowing for negative polarity values, which makes it easier to visualize the raw event data. This is different from other interpretations in the literature, which discard event polarity information.

## Conventional Image Denoising Techniques

Bilateral filtering, while inefficient for large kernel sizes, 


# References

E. Mueggler, H. Rebecq, G. Gallego, T. Delbruck, D. Scaramuzza
The Event-Camera Dataset and Simulator: Event-based Data for Pose Estimation, Visual Odometry, and SLAM
International Journal of Robotics Research, Vol. 36, Issue 2, pages 142-149, Feb. 2017.