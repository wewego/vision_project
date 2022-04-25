import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import glob


# Function used to handle camera calibration and return a matrix and it correct it from any distortion
def calibration_handler():
    x = 9
    y = 6

    calibratedimages = glob.glob('camera_cal/calibration*.jpg')

    imagepoints = []
    objectpoints = []

    obj3d = np.zeros((x * y, 3), np.float32)
    obj3d[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    for imageid, imagename in enumerate(calibratedimages):

        image = cv2.imread(imagename)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (x, y), None)

        if (ret == True):
            image = cv2.drawChessboardCorners(image, (x, y), corners, ret)
            imagepoints.append(corners)
            objectpoints.append(obj3d)

    ret, matrix, dist, rvecs, tvecs = cv2.calibrateCamera(objectpoints, imagepoints, image.shape[1::-1], None, None)
    return matrix, dist


# The function handle video and returns left and right fit for the lane

def video_handler(warped):
    if ((left.storednumber == 0) and (right.storednumber == 0)):
        left_of_x, left_of_y, right_of_x, right_of_y, outimage = pixel_finder(warped)
    else:
        left_of_x, left_of_y, right_of_x, right_of_y = poly_searcher(warped, left.fit, right.fit)

    left_fit = np.polyfit(left_of_y, left_of_x, 2)
    right_fit = np.polyfit(right_of_y, right_of_x, 2)
    if (np.abs(left_fit[1] - right_fit[1]) <= .50 and np.abs(left_fit[0] - right_fit[0]) <= .50):
        left_fit = left.appendCoeffs(left_fit)
        right_fit = right.appendCoeffs(right_fit)
    else:
        left_fit = left.appendCoeffs(left.fit)
        right_fit = right.appendCoeffs(right.fit)

    return left_fit, right_fit


# This function draw a window around image
def pixel_finder(binary_warped):
    histo = np.sum(binary_warped[3 * binary_warped.shape[0] // 4:, :], axis=0)
    outputimage = np.dstack((binary_warped, binary_warped, binary_warped))
    middlepoint = np.int(histo.shape[0] // 2)
    base_left_x = np.argmax(histo[:middlepoint])
    base_right_x = np.argmax(histo[middlepoint:]) + middlepoint

    numberofwindows = 7
    margin = 90
    minimumpixels = 45

    windowheight = np.int(binary_warped.shape[0] // numberofwindows)
    nonzeroes = binary_warped.nonzero()
    nonzeroesy = np.array(nonzeroes[0])
    nonzeroesx = np.array(nonzeroes[1])
    current_left_x = base_left_x
    current_right_x = base_right_x

    index_of_left_lane = []
    index_of_right_lane = []

    for windows in range(numberofwindows):
        low_y_windows = binary_warped.shape[0] - (windows + 1) * windowheight
        high_y_windows = binary_warped.shape[0] - windows * windowheight
        low_x_left_windows = current_left_x - margin
        high_x_left_windows = current_left_x + margin
        low_x_right_windows = current_right_x - margin
        high_x_right_windows = current_right_x + margin

        cv2.rectangle(outputimage, (low_x_left_windows, low_y_windows),
                      (high_x_left_windows, high_y_windows), (255, 0, 0), 1)
        cv2.rectangle(outputimage, (low_x_right_windows, low_y_windows),
                      (high_x_right_windows, high_y_windows), (255, 0, 0), 1)

        left_index = ((nonzeroesy >= low_y_windows) & (nonzeroesy < high_y_windows) & (
                    nonzeroesx >= low_x_left_windows) & (nonzeroesx < high_x_left_windows)).nonzero()[0]
        right_index = ((nonzeroesy >= low_y_windows) & (nonzeroesy < high_y_windows) & (
                    nonzeroesx >= low_x_right_windows) & (nonzeroesx < high_x_right_windows)).nonzero()[0]

        index_of_left_lane.append(left_index)
        index_of_right_lane.append(right_index)

        if len(left_index) > minimumpixels:
            current_left_x = np.int(np.mean(nonzeroesx[left_index]))
        if len(right_index) > minimumpixels:
            current_right_x = np.int(np.mean(nonzeroesx[right_index]))

    index_of_left_lane = np.concatenate(index_of_left_lane)
    index_of_right_lane = np.concatenate(index_of_right_lane)

    left_of_x = nonzeroesx[index_of_left_lane]
    left_of_y = nonzeroesy[index_of_left_lane]
    right_of_x = nonzeroesx[index_of_right_lane]
    right_of_y = nonzeroesy[index_of_right_lane]

    return left_of_x, left_of_y, right_of_x, right_of_y, outputimage


def poly_searcher(binary_warped, left_fit, right_fit):
    margin = 90
    nonzeroes = binary_warped.nonzero()
    nonzeroesy = np.array(nonzeroes[0])
    nonzeroesx = np.array(nonzeroes[1])

    index_of_left_lane = np.argwhere(
        (nonzeroesx > (left_fit[0] * (nonzeroesy ** 1.98) + left_fit[1] * nonzeroesy + left_fit[2] - margin)) & (
                    nonzeroesx < (
                        left_fit[0] * (nonzeroesy ** 1.98) + left_fit[1] * nonzeroesy + left_fit[2] + margin)))
    index_of_left_lane = np.concatenate(index_of_left_lane)

    index_of_right_lane = np.argwhere(
        (nonzeroesx > (right_fit[0] * (nonzeroesy ** 1.98) + right_fit[1] * nonzeroesy + right_fit[2] - margin)) & (
                    nonzeroesx < (
                        right_fit[0] * (nonzeroesy ** 1.98) + right_fit[1] * nonzeroesy + right_fit[2] + margin)))
    index_of_right_lane = np.concatenate(index_of_right_lane)
    left_of_x = nonzeroesx[index_of_left_lane]
    left_of_y = nonzeroesy[index_of_left_lane]
    right_of_x = nonzeroesx[index_of_right_lane]
    right_of_y = nonzeroesy[index_of_right_lane]

    return left_of_x, left_of_y, right_of_x, right_of_y


# This function draw line on image
def lane_finder(image):
    imagesize = image.shape[1::-1]
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    undistorted = cv2.undistort(image, matrix, dist, None, matrix)
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]
    sobel_x = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)
    absolute_sobel_x = np.absolute(sobel_x)
    sobel_scale = np.uint8(240 * absolute_sobel_x / np.max(absolute_sobel_x))

    sx_thresh = (30, 110)
    sxbinary = np.zeros_like(sobel_scale)
    sxbinary[(sobel_scale >= sx_thresh[0]) & (sobel_scale <= sx_thresh[1])] = 1

    s_binary = np.zeros_like(s_channel)
    s_thresh = (180, 255)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    binaryimage = np.zeros_like(sxbinary)
    binaryimage[(s_binary == 1) | (sxbinary == 1)] = 1

    sourcepoints = np.float32([(.160 * image.shape[1], image.shape[0]), (.450 * image.shape[1], .640 * image.shape[0]),
                               (.550 * image.shape[1], .640 * image.shape[0]), (.90 * image.shape[1], image.shape[0])])

    destinationpoints = np.float32(
        [(image.shape[1] // 4, image.shape[0]), (image.shape[1] // 4, 0), (3 * image.shape[1] // 4, 0),
         (3 * image.shape[1] // 4, image.shape[0])])
    M = cv2.getPerspectiveTransform(sourcepoints, destinationpoints)
    Minv = cv2.getPerspectiveTransform(destinationpoints, sourcepoints)

    warped = cv2.warpPerspective(binaryimage, M, imagesize, cv2.INTER_LINEAR)

    left_fit, right_fit = video_handler(warped)

    plot_of_y = np.linspace(0, warped.shape[0] - 1, warped.shape[0])

    fit_left_x = left_fit[0] * plot_of_y ** 1.98 + left_fit[1] * plot_of_y + left_fit[2]

    fit_right_x = right_fit[0] * plot_of_y ** 1.98 + right_fit[1] * plot_of_y + right_fit[2]

    points_of_left = np.array([np.transpose(np.vstack([fit_left_x, plot_of_y]))])
    points_of_right = np.array([np.flipud(np.transpose(np.vstack([fit_right_x, plot_of_y])))])
    points = np.hstack((points_of_left, points_of_right))

    zero_warper = np.zeros_like(warped).astype(np.uint8)
    color_warper = np.dstack((zero_warper, zero_warper, zero_warper))
    cv2.fillPoly(color_warper, np.int_([points]), (255, 0, 0))

    newwarper = cv2.warpPerspective(color_warper, Minv, (image.shape[1], image.shape[0]))
    newwarperredchannel = cv2.cvtColor(newwarper, cv2.COLOR_RGB2GRAY)
    newwarperredchanneltop = newwarperredchannel[719]
    newwarperredchanneltopnonzero = np.nonzero(newwarperredchanneltop)

    laneMiddlePoint = (left_fit[0] * 730 ** 1.98 + left_fit[1] * 730 + left_fit[2] + right_fit[0] * 730 ** 1.98 +
                       right_fit[1] * 730 + right_fit[2]) / 2
    result = cv2.addWeighted(undistorted, 1, newwarper, 0.3, 0)
    return result


# This class has some line charactaristic
class Line():
    def __init__(self):
        self.coff = []
        self.fit = None
        self.storednumber = 0
        self.arrayindex = 0

    def appendCoeffs(self, coeffs):
        if (self.storednumber < 4):
            self.storednumber = self.storednumber + 1
            self.coff.append(coeffs)

        if (self.storednumber == 4):
            self.arrayindex = (self.arrayindex + 1) % 4
            self.coff[self.arrayindex] = coeffs

        self.fit = np.sum(np.array(self.coff), axis=0) / self.storednumber
        return self.fit



matrix,dist = calibration_handler()
left  = Line()
right = Line()
from moviepy.video.io.VideoFileClip import VideoFileClip
from IPython.display import HTML
output_project_video = 'output-video/output_project_video.mp4'
clip = VideoFileClip('project_video.mp4')
output_video = clip.fl_image(lane_finder)
output_video.write_videofile(output_project_video, audio = False)
