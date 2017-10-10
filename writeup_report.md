**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undistort_camera_cal.jpg "Undistorted chessboard image"
[image11]: ./camera_cal/calibration1.jpg "Distorted chessboard image"
[image2]: ./test_images/test2.jpg "Distorted test image"
[image21]: ./output_images/w_test2.jpg "Undistorted test image"
[image3]: ./output_images/threshold_gradient.jpg "Threshold_gradient"
[image4]: ./output_images/persp_transform.png "Perspective Transform"
[image5]: ./output_images/polyfit.png "Polyfit"
[image6]: ./output_images/final_image.jpg "Output"
[video1]: ./out_project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the function get_dist_mtx() in the file project_utils.py  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image11]  ![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images:

Once having found the distortion coefficients and the camera matrix through the camera calibration process, these are used using the undistort function in opencv on the test image.The distorted and undistorted test images are shown below
![alt text][image2]  ![alt text][image21]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (code is in functions pipeline() and dir_threshold() in the file project_utils.py.  Here's an example of my output for this step.  

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform is in lines 36-46 of the process_image() function in advlanelines.py.  I chose to hardcode the source and destination points based on an image with straight lane lines in the following manner:

```python
src = np.float32(
    [[577, 460],[200, 720],[1110, 720],[702, 460]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Once the perspective transform was obtained through a straight lines image, I applied the transform to all the binary images obtained by applying color and gradinet thresholds to the test images. A histogram of the binary pixels in the x direction was obtained to get a rough idea of the left and right lanes. 
A sliding window search was then conducted around this rough position to identify non zero pixels that correspond to the lanes. A second order polynomial is then fit to the non zero pixels within the right and left lanes.
The code is in the findLanes() function in project_utils.py

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The radius of curvature was found as suggested in the class. The code is in lines 240-245 of the findLanes() function in project_utils.py. The position of the vehicle w.r.t. lane center is calculated in Lines 60-65 of advlanelines.py

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

This in lines 69-85 of advlanelines.py. The final result is shown below.

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

For the video, I mostly used the same pipeline that I had for the images. In addition, I had the following enhancements
* To reduce wobbliness as well as to cover for frames where I did not obtain good lane identification, I used smoothing of the lanes across frames with valid identification.
* I also had checks on the results within each frame based on if the lanes were reasonably parallel and if they yielded reasonable radius of curvature.

Here's a [link to my video result](./out_project_video.mp4)

![alt text][video1]
---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Among the various ideas discussed in the lectures to obtain a thesholded binary image, I used a x gradient check on the L channel, a range check on the S- channel level and thresholding of the gradient directions. This was able to identify the lane lines reasonably well but in addition, it identified a few other lines corresponding to edges of medians. So I had to further constrain the search windows to the middle of the image to reject a few other edges.

My pipeline seems to find it tough to identify the lanes under lightly colored surfaces and under certain shadows. This as not an issue for the project video but it was an issue for the challenge videos. Other ideas to make th epipeline more robust could include exploring other color spaces and more intelligent rejection and filtering of the binary image. 
