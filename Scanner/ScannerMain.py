'''
Author: Pridhvi Myneni

Scanner Application
Instructions: 
1) Set the correct paths for image input and output
2) If running on a mac, change the start command to run, or whatever macs use to open a file (you can comment it out, if you don't want the script to auto-open the file when it's done running
'''

import os  # This import is not required, and is only being used for my own convenience while testing. 

import cv2  # Actual necessary import

import ImageProcessor  # This is the other file in the same directory: it's the library that I'm developing for my own use
import numpy as np  # ehh, I don't know why I imported numpy, but I feel like I'm going to need it

'''
I need to make this easier for the user to set the path, but for right now, it's just for my convenience. 
Returns the image, as provided in the path, which is hard coded below
'''


def getImage():
    return cv2.imread("C:/users/bluer/Desktop/WIN_20171114_13_48_53_Pro.jpg")


'''
This is not currently in use, I think. This was designed to make computations faster when I'm experimenting with automatic boundry-settings (for canny)
Resizes the image such that it maintains its aspect ratio, but its height and width are less than or equal to 500 px
'''


def resizeImage(img):
    (row, col, chan) = img.shape  # get the shape of the image. Channel isn't being used, but I figured it would be useful to keep
    while row > 500 or col > 500:  # Basically, just keep dividing by 2 until the width/height are small enough
        # Yes, I know this is really stupid code and it can be simplified by just dividing row and col by 500 and rounding up
        # But, it's not in use right now, and I'll make that change later (I'm only documenting right now)
        row = row / 2
        col = col / 2
    another = img.copy()
    another = cv2.resize(another, (row, col))
    img = another
    (row, col, chan) = img.shape
    cv2.imshow("", another)  # Display the image, so I know that my horrible complex resizing code worked... yes, I know it needs to be fixed
    cv2.waitKey()
    return another


'''
Main Logic
'''
img = getImage()  # Get the image
edges = ImageProcessor.edges(img, 50, 175, 7)  # Using settings that are HARDCODED, the library is returning the edges
cv2.imshow("", edges)  # it's showing the image, so (1) I know I have the right picture and (2) if/when the script breaks, I might be able to use this to help me debug
cv2.waitKey()
cv2.destroyAllWindows()
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
(contoursFound, _) = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
'''
As you can tell from the following command, I take the contours and sort them by their contour area...
'''
contoursFound = sorted(contoursFound, key=cv2.contourArea, reverse=True)[:10]
screenCnt = contoursFound[0]  # Just initializing the selected one to the contour with the largest area. 
cnthold = []
cnter = 0

for c in contoursFound:
    cnter += 1
    contourPerimeter = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * contourPerimeter, True)  # 0.02 is just an arbitrary value, which specifies the tolerance the contour can have to approximate itself, for compression purposes
    cnthold.append((c, contourPerimeter))  # Basically store the contour, along with its perimeter

dist = []

for i in cnthold:
    # dist.append(abs(4-i[1])) #This was originally based on perimeter, but now, as you can see from the next line...
    dist.append(cv2.contourArea(i[0]))  # It's all based on the area...
absmin = max(dist)  # this was absmin because it was based on number of vertices but now it's based on the area, which we want the most of
ind = dist.index(absmin)#the index of the array that contains the max area... yes, there's probably a cleaner&easier way to do this
screenCnt = cnthold[ind][0] #Final contour = the one with most area
cv2.drawContours(img, [screenCnt], -1, (0, 255, 0), 2) #Draw the contour on the image so I know if it worked
ImageProcessor.saveImage(img, "C:/users/bluer/Desktop/reco2.jpg")#Save the output image
os.system("start " + "C:/users/bluer/Desktop/reco2.jpg")#Open it
'''
Closing thoughts: I should probably add most of what's in my main method to the ImageProcessor file so it's part of my library (for contours), but that's out of the question until I can figure out how to do this without hardcoding in parameters for the edge function (see laplacian algorithm comment in email)
'''
