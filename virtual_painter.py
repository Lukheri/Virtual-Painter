import cv2
import numpy as np
import mediapipe as mp
import HandTrackingModule as htm
import math

class ColorChanger:
    color_pointer = 0
    red = (0, 0, 255)
    green = (0, 255, 0)
    purple = (255, 0, 255)
    yellow = (0, 255, 255)
    brown = (0, 52, 102)
    colors = [red, green, purple, yellow, brown]
    change_color = False

    def get_color(self):
        return self.colors[self.color_pointer]
    
    def get_change_color(self):
        return self.change_color
    
    def set_color_pointer(self):
        if self.color_pointer >= len(self.colors)-1:
            self.color_pointer = 0
        else:
            self.color_pointer += 1

    def set_change_color_true(self):
        self.change_color = True

    def set_change_color_false(self):
        self.change_color = False

class PrevCoordinates:
    prev_x = 0
    prev_y = 0

    def get_prev_xy(self):
        return (self.prev_x, self.prev_y)
    
    def set_prev_xy(self, x, y):
        self.prev_x, self.prev_y = x, y
        
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = htm.handDetector(detectionCon=0.85)

imgCanvas = np.zeros((720, 1280, 3), np.uint8)

color = ColorChanger()
drawing_coordinates = PrevCoordinates()
eraser_coordinates = PrevCoordinates()

def draw_circle_eraser(x1, x2, y1, y2):
    radius = int(math.dist([x1, y1], [x2, y2])/2)
    centerx = (x1 + x2)//2
    centery = (y1 + y2)//2
    center = (centerx, centery)
    return center, radius

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]
        x3, y3 = lmList[20][1:]
    
        fingers = detector.fingersUp()
        center, radius = draw_circle_eraser(x1, x2, y1, y2)

        if not fingers[0] and fingers[1] and fingers[2] and not fingers[3]:
            if not eraser_coordinates.prev_x and not eraser_coordinates.prev_y:
                eraser_coordinates.set_prev_xy(*center)

            cv2.circle(img, center, radius+15, (0, 0, 0), cv2.FILLED)
            cv2.line(imgCanvas, eraser_coordinates.get_prev_xy(), center, (0,0,0), (radius+15)*2)
        
        if not fingers[0] and fingers[1] and not fingers[2] and not fingers[3]:
            if not drawing_coordinates.prev_x and not drawing_coordinates.prev_y:
                drawing_coordinates.set_prev_xy(x1, y1)

            cv2.circle(img, (x1, y1), 15, color.get_color(), cv2.FILLED)
            cv2.line(imgCanvas, drawing_coordinates.get_prev_xy(), (x1, y1), color.get_color(), 15)
        
        if not fingers[2] and not fingers[3] and fingers[4]:
            cv2.circle(img, (x3, y3), 7, color.get_color(), cv2.FILLED)
            if color.get_change_color():
                color.set_color_pointer()
                color.set_change_color_false()
                
        if not fingers[4]:
            color.set_change_color_true()
            
        drawing_coordinates.set_prev_xy(x1, y1)
        eraser_coordinates.set_prev_xy(*center)

    # Makes you draw on the video
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img,imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    cv2.imshow("Image", img)
    cv2.waitKey(1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break