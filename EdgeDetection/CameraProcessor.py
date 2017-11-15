'''
Author: Pridhvi Myneni
Version: Python 2.7.13

Instructions:
1) Read all the instructions before you do anything
2) Change the preferences below to match your personal preferences and the machine on which you will be running the script
3) Run the program
4) Use 'q' and 'e' keys to play with the gaussian blur settings: 'q' makes it go less (starting at the minimum, so it won't go any further) and 'e' makes it blur more.
5) Use 'w', 'a', 's', 'd' keys to play with the boundaries for the Canny algorithm. Use 'a' and 'd' to modify the lower boundary, and 'w' and 's' to modify the upper boundary. 
6) This is a continuation of 5, but I ran out of space: 'a' and 's' are for making the value smaller, 'd' and 'w' are for making it bigger.  
'''
import cv2 #Yes, you need OpenCV Installed

# Constants to be initialized based on the user's machine and preferences
WIDTH = 1280  # Width of the display
HEIGHT = 720  # Height of the display
DIMENSIONS = (WIDTH, HEIGHT)  # Making a tuple because I'm lazy (hint: not a setting to be changed)
INPUTdEVICEnUMBER = 0  # If your device has multiple camera interfaces, use this to select which one you would like to select

'''
This method exists to return the inverted color version of the image submitted. 
Note that this method only works SUCCESSFULLY on black and white images. Grayscale not supported.
'''


def invertBlackAndWhite(img):
    inverted = cv2.bitwise_not(img, 1)  # could be 1 or 0, doesn't make a difference
    return inverted


'''
This method returns the contours of a given image, after taking a gaussian blur, to remove image noise.
The user, through his/her key inputs can select the gaussian blur settings and the contour detection method parameters
'''


def contoursOnly(image, boundryLower, boundryUpper, num):
    noiselessImage = cv2.GaussianBlur(image, (num, num), 0) #Use a gaussian blur to get rid of image noise, which will become annoying in the next step
    contours = cv2.Canny(noiselessImage, boundryLower, boundryUpper) #Get the contours of the img using Canny's algorithm
    return contours


'''
This method shows the image to the user, through the cv2 library
'''


def displayImage(img, windowName=""):
    cv2.imshow(windowName, cv2.resize(img, DIMENSIONS))

'''
Main Logic of Program
'''
cap = cv2.VideoCapture(INPUTdEVICEnUMBER) #Get the camera
boundryUpper = 35 #Set the initial boundaries, Note that these are numbers
boundryLower = 10 #that I've found work well for my system. 
num = 1 #Number for Gaussian Blur (must be >0). At num=1, there is no smoothing. At greater values, you get more smoothing, though this isn't necessarily great, as it can result in the edges becoming obscured or not detected
print("Camera has been successfully initialized.")  # Debugging statement, as some computers may hang on the initialization part, or outright fail when attempting to use the camera, if none is attached
while True:
        ret, img = cap.read()#ret = whether or not a frame was returned, img is the frame taken at the instant the method was called
        contours = contoursOnly(img, boundryLower, boundryUpper, num) #Get contours
        displayImage(invertBlackAndWhite(contours)) #Invert the image
        key = cv2.waitKey(1)  # set to 0, if you don't want a stream, but rather an image at a time. If the machine is running slowly, increase the wait time (in milliseconds)
        if key == 27: #escape key to exit
            break
        if key == 119:#See instructions above program for what these do
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
        if key == 113 and num > 1: #If num<=1, then gaussian blur breaks. It also has to be odd, or else it will also break (hence, multiples of 2 added to 1). 
            num -= 2
        if key == 101:
            num += 2
print boundryUpper
print boundryLower
print num
#When the user quits, the program removes all the windows and releases the resources (camera) so other applications can use them.
cv2.destroyAllWindows()
cv2.VideoCapture(0).release()
