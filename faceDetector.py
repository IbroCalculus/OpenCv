import cv2
import pyautogui as pya

screen_resolution = pya.size()
screen_width = screen_resolution[0]
screen_height = screen_resolution[1]

face_cascade = cv2.CascadeClassifier("cascades/data/haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("cascades/data/haarcascade_eye.xml")

# Default webcam, for both pictures and videos
video = cv2.VideoCapture(0)
'''
Resolutions

video.set(3, 1920) #3,width
video.set(4, 1080) #4,height

'''

def R_1080p():
    video.set(3, 1920)
    video.set(4, 1080)

def R_720p():
    video.set(3, 1280)
    video.set(4, 720)
    
def R_480p():
    video.set(3, 640)
    video.set(4, 480)

def R_custom(width, height):
    video.set(3, width)
    video.set(4, height)


R_custom((screen_width/2), (screen_height/2))


while True:
    check,frame = video.read()
 
    gray_video = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #search face coordinates of the grayscale image
    faces = face_cascade.detectMultiScale(gray_video, scaleFactor = 1.05,
                                          minNeighbors=5)
    eyes = eye_cascade.detectMultiScale(gray_video, scaleFactor = 1.05,
                                          minNeighbors=5)
    #Adding rectangle with thickness, i.e 1 and 10
    for x,y,w,h in eyes:
        rect_indicator = cv2.rectangle(frame, (x,y), (x+w,y+h),(0,255,0),1)

    for x,y,w,h in faces:
        rect_indicator = cv2.rectangle(gray_video, (x,y), (x+w,y+h),(0,255,0),10)

    cv2.imshow("Gray",gray_video)
    cv2.imshow("Colored",frame)
    key = cv2.waitKey(1)
    '''
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    '''
    if key == ord('q'):
        break
video.release()
cv2.destroyAllWindows()
