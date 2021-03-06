'''
Author: Pridhvi Myneni

Scanner Application
Instructions:
1) Set the correct paths for image input and output
2) If running on a mac, change the start command to run, or whatever macs use to open a file (you can comment it out, if you don't want the script to auto-open the file when it's done running
'''

# This import is not required, and is only being used for my own convenience while testing.
import os

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
    return cv2.imread("C:/users/bluer/Desktop/reco3.jpg")


'''
This is not currently in use, I think. This was designed to make computations faster when I'm experimenting with automatic boundry-settings (for canny)
Resizes the image such that it maintains its aspect ratio, but its height and width are less than or equal to 500 px
'''


def resizeImage(img):
    # get the shape of the image. Channel isn't being used, but I figured it would be useful to keep
    (row, col, chan) = img.shape
    while row > 500 or col > 500:  # Basically, just keep dividing by 2 until the width/height are small enough
        # Yes, I know this is really stupid code and it can be simplified by just dividing row and col by 500 and rounding up
        # But, it's not in use right now, and I'll make that change later (I'm only documenting right now)
        row = row / 2
        col = col / 2
    another = img.copy()
    another = cv2.resize(another, (row, col))
    img = another
    (row, col, chan) = img.shape
    # Display the image, so I know that my horrible complex resizing code worked... yes, I know it needs to be fixed
    cv2.imshow("", another)
    cv2.waitKey()
    return another


'''
Main Logic
'''
# cap = cv2.VideoCapture(0)  # Get the camera
boundryLower = 0
boundryUpper = 50
num = 20
num1 = 10
num2 = 10
while(True):
    #ret, img = cap.read()
    img = getImage()
    # Using settings that are HARDCODED, the library is returning the edges
    edges = ImageProcessor.bilateralFilterEdges(
        img, boundryLower, boundryUpper, num, num1, num2)
    # it's showing the image, so (1) I know I have the right picture and (2) if/when the script breaks, I might be able to use this to help me debug
    cv2.imshow("", edges)
    # cv2.destroyAllWindows()
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
    #(contoursFound, _) = cv2.findContours(
#        edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    '''
    As you can tell from the following command, I take the contours and sort them by their contour area...
    '''
    '''
    contoursFound = sorted(
        contoursFound, key=cv2.contourArea, reverse=True)[:10]
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

    dist = []

    for i in cnthold:
        # dist.append(abs(4-i[1])) #This was originally based on perimeter, but now, as you can see from the next line...
        dist.append(cv2.contourArea(i[0]))  # It's all based on the area...
    # this was absmin because it was based on number of vertices but now it's based on the area, which we want the most of
    absmin = max(dist)
    # the index of the array that contains the max area... yes, there's probably a cleaner&easier way to do this
    ind = dist.index(absmin)
    # screenCnt = cnthold[ind][0]  # Final contour = the one with most area
    # Draw the contour on the image so I know if it worked
    #cv2.drawContours(img, [screenCnt], -1, (0, 255, 0), 2)
    #cv2.imshow("1", img)
    # set to 0, if you don't want a stream, but rather an image at a time. If the machine is running slowly, increase the wait time (in milliseconds)
    '''
    key = cv2.waitKey(1)
    if key == 27:  # escape key to exit
        break
    if key == 119:  # See instructions above program for what these do
        # up
        boundryUpper += 10
    if key == 115:
        # down
        boundryUpper -= 10
    if key == 97:
        # down
        boundryLower -= 10
    if key == 100:
        # up
        boundryLower += 10
    # If num<=1, then gaussian blur breaks. It also has to be odd, or else it will also break (hence, multiples of 2 added to 1).
    if key == 113 and num > 1:
        num -= 50
    if key == 101:
        num += 50
    if key == 49:
        num1 -= 50
    if key == 50:
        num1 += 50
    if key == 52:
        num2 -= 50
    if key == 53:
        num2 += 50
print boundryLower
print boundryUpper
print num
print num1
print num2
# ImageProcessor.saveImage(img, "C:/users/bluer/Desktop/reco2.jpg")#Save the output image
# os.system("start " + "C:/users/bluer/Desktop/reco2.jpg")#Open it
'''
Closing thoughts: I should probably add most of what's in my main method to the ImageProcessor file so it's part of my library (for contours), but that's out of the question until I can figure out how to do this without hardcoding in parameters for the edge function (see laplacian algorithm comment in email)
'''
cv2.destroyAllWindows()
cap.release()
