# Directed Project: Optimization of Tracking Framework

Contact: david.moreno@ieee.org

Faster computation of [HOG features](https://en.wikipedia.org/wiki/Histogram_of_oriented_gradients) for object tracking using Matlab and OpenCV with a mex interface. For a deteails about the study, please see [this file](pd-report.pdf). Any feedback is immensely appreciated!

The original tracking framework used [Matlab's implementation of HOG features](https://www.mathworks.com/help/vision/ref/extracthogfeatures.html), which resulted in FPS in the range of 0.35 to 1, whereas optimized implementation achieves FPS that range between 1.39 and 12.89. This was accomplished with the help of [OpenCV's library](https://docs.opencv.org/3.2.0/d5/d33/structcv_1_1HOGDescriptor.html) and [Matlab's mex interface](https://www.mathworks.com/discovery/matlab-opencv.html).

## Requirements
Matlab and OpenCV: https://www.mathworks.com/discovery/matlab-opencv.html

## How to use

First, call the class constructor:
```hogObj = HOGDescriptor(size(image),'CellSize',CellSize,'BlockSize',BlockSize,'BlockOverlap',BlockOverlap);```
Then, compute the features with:
```featFst = compute(hogObj,image);```
When done, release the object:
```release(hogObj);```
