'''
Author: Pridhvi Myneni
test
Scanner Application
Instructions:
1) Set the correct paths for image input and output
2) If running on a mac, change the start command to run, or whatever macs use to open a file (you can comment it out, if you don't want the script to auto-open the file when it's done running
'''
print
# This import is not required, and is only being used for my own convenience whilse testing.
import os
import numpy
import time


import cv2  # Actual necessary import

# This is the other file in the same directory: it's the library that I'm developing for my own use
import ImageProcessor
# ehh, I don't know why I imported numpy, but I feel like I'm going to need it
import numpy as np

'''
I need to make this easier for the user to set the path, but for right now, it's just for my convenience.
Returns the image, as provided in the path, which is hard coded below
'''


def getImage():
    return cv2.imread("C:/users/bluer/Desktop/Input (3).jpg")


'''
This is not currently in use, I think. This was designed to make computations faster when I'm experimenting with automatic boundry-settings (for canny)
Resizes the image such that it maintains its aspect ratio, but its height and width are less than or equal to 500 px
'''


def resizeImage(img):
    # get the shape of the image. Channel isn't being used, but I figured it would be useful to keep
    shit = img.shape
    row, col = shit[0], shit[1]
    print row, col
    while row > 750 or col > 750:  # Basically, just keep dividing by 2 until the width/height are small enough
        # Yes, I know this is really stupid code and it can be simplified by just dividing row and col by 500 and rounding up
        # But, it's not in use right now, and I'll make that change later (I'm only documenting right now)
        row = row / 2
        col = col / 2
    another = img.copy()
    another = cv2.resize(another, (row, col))
    #img = another
    #(row, col, chan) = img.shape
    # Display the image, so I know that my horrible complex resizing code worked... yes, I know it needs to be fixed
    cv2.imshow("", another)
    cv2.waitKey()
    cv2.destroyAllWindows()
    return another


def corners(arr):
    # upRight
    maxSum = -1
    saveMe = None
    for i in arr:
        tsum = i[0][0] + i[0][1]
        if tsum > maxSum:
            maxSum = tsum
            saveMe = i
    # downLeft
    height, width = getImage().shape[:2]
    minsum = height + width + 1
    saveMeLeftBottom = None
    for i in arr:
        tsum = i[0][0] + i[0][1]
        if tsum < minsum:
            minsum = tsum
            saveMeLeftBottom = i
    # UpLeft
    mindistleft = height + width + 1
    saveMeTopLeft = None
    for i in arr:
        td = pow(pow(i[0][0] - width, 2) + pow(i[0][1], 2), 0.5)  # 0,0
        if td < mindistleft:
            mindistleft = td
            saveMeTopLeft = i
    # downRight
    mindistright = height + width + 1
    saveMeRight = None
    for i in arr:
        td = pow(pow(i[0][0], 2) + pow(height - i[0][1], 2), 0.5)
        if td < mindistright:
            mindistright = td
            saveMeRight = i
    retArr = [saveMe[0], saveMeTopLeft[0],
              saveMeLeftBottom[0], saveMeRight[0]]
    return retArr


def four_point_transform(image, pts):
        # obtain a consistent order of the points and unpack them
        # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped


def order_points(pts):
        # initialzie a list of coordinates that will be ordered
        # such that the first entry in the list is the top-left,
        # the second entry is the top-right, the third is the
        # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect


'''
Main Logic
'''
img = getImage()  # Get the image
# Using settings that are HARDCODED, the library is returning the edges
edges = cv2.GaussianBlur(img, (15, 15), 0)
edges = cv2.bilateralFilter(edges, 10, 10, 0)
resizeImage(edges)
edges = cv2.Laplacian(edges, cv2.CV_32F)
edges = np.uint8(edges)
ImageProcessor.saveImage(edges, "C:/users/bluer/Desktop/reco3.jpg")
# This line only change the type, not values
resizeImage(edges)
edges = ImageProcessor.medianBlurEdges(edges, 10, 30, 5)
resizeImage(edges)
edges = cv2.Laplacian(edges, cv2.CV_32F)
edges = np.uint8(edges)
resizeImage(edges)
#edges = ImageProcessor.bilateralFilterEdges(edges, 100, 300, 50, 50, 1)
resizeImage(edges)
# it's showing the image, so (1) I know I have the right picture and (2) if/when the script breaks, I might be able to use this to help me debug
#cv2.imshow("", edges)
'''
Actual work below
'''
'''
There's a lot of stuff going on in the next line, so I'm making a multiline comment:
    - so, we're using edges.copy() because apparently findContours modifies the original image
    - cv2.RETR_LIST specifies that opencv shouldn't bother with heirarchical stuff, which
        is why we don't care about the other output of the command (_)
    - cv2.CHAIN_APPROX_SIMPLE specifies that the returned contours should be compressed by only having
        their endpoints stored for diag/hor/vert segments, as opposed to storing every single point

'''
(contoursFound, _) = cv2.findContours(
    edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
'''
As you can tell from the following command, I take the contours and sort them by their contour area...
'''
contoursFound = sorted(contoursFound, key=cv2.contourArea, reverse=True)[:10]
# Just initializing the selected one to the contour with the largest area.
screenCnt = contoursFound[0]
cnthold = []
cnter = 0

for c in contoursFound:
    cnter += 1
    contourPerimeter = cv2.arcLength(c, True)
    # 0.02 is just an arbitrary value, which specifies the tolerance the contour can have to approximate itself, for compression purposes
    approx = cv2.approxPolyDP(c, 0.02 * contourPerimeter, True)
    # Basically store the contour, along with its perimeter
    cnthold.append((c, contourPerimeter))
    # cv2.drawContours(img, [c], -1, (0, 255, 0), 2)  # screenCnt

dist = []

for i in cnthold:
    # dist.append(abs(4-i[1])) #This was originally based on perimeter, but now, as you can see from the next line...
    dist.append(cv2.contourArea(i[0]))  # It's all based on the area...
# this was absmin because it was based on number of vertices but now it's based on the area, which we want the most of
absmin = max(dist)
# the index of the array that contains the max area... yes, there's probably a cleaner&easier way to do this
ind = dist.index(absmin)
screenCnt = cnthold[ind][0]  # [0]  # Final contour = the one with most area
# Draw the contour on the image so I know if it worked
# readd brackets around screenCnt
# cv2.drawContours(img, [screenCnt], -1, (0, 255, 0), 2)  # screenCnt
# apply the four point transform to obtain a top-down
# view of the original image
#----------
# print screenCnt
# print len(screenCnt)
fourcount = corners(screenCnt)
width, height = getImage().shape[:2]
p1 = numpy.float32(fourcount)
ctr = numpy.array(p1).reshape((-1, 1, 2)).astype(numpy.int32)
cntllr = [ctr[0], ctr[1], ctr[2], ctr[3]]
print height, width
p1 = cntllr
p2 = numpy.array([[2048, 1536], [2048, 0], [0, 0], [0, 1536]])
ctr = numpy.array(p2).reshape((-1, 1, 2)).astype(numpy.int32)
cntllr = [ctr[0], ctr[1], ctr[2], ctr[3]]
p2 = cntllr
M = cv2.getPerspectiveTransform(np.float32(p1), np.float32(p2))
# warped = four_point_transform(img, fourcount.reshape(4, 2) * ratio)
warped = cv2.warpPerspective(img, M, (2048, 1536))
# convert the warped image to grayscale, then threshold it
# to give it that 'black and white' paper effect
# warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
#------------
# Save the output image


#cv2.drawContours(img, p2, -1, (255, 0, 0), 50)
cv2.drawContours(img, p1, -1, (0, 0, 255), 50)
ImageProcessor.saveImage(img, "C:/users/bluer/Desktop/reco1.jpg")
ImageProcessor.saveImage(warped, "C:/users/bluer/Desktop/reco2.jpg")
os.system("start " + "C:/users/bluer/Desktop/reco2.jpg")  # Open it
os.system("start " + "C:/users/bluer/Desktop/reco1.jpg")  # Open it
'''
Closing thoughts: I should probably add most of what's in my main method to the ImageProcessor file so it's part of my library (for contours), but that's out of the question until I can figure out how to do this without hardcoding in parameters for the edge function (see laplacian algorithm comment in email)
'''
