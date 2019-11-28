import cv2

cascade_src1 = '/home/swati/Desktop/head.xml'
cascade_src = '/home/swati/Desktop/two_wheeler.xml'
video_src = '/home/swati/Desktop/bikes.mp4'

cap = cv2.VideoCapture(video_src)
fgbg = cv2.createBackgroundSubtractorMOG2()
car_cascade = cv2.CascadeClassifier(cascade_src)
car_cascade1 = cv2.CascadeClassifier(cascade_src1)

while True:
    ret, img = cap.read()
    fgbg.apply(img)
    if (type(img) == type(None)):
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cars = car_cascade.detectMultiScale(gray, 1.1, 1)
    cars1 = car_cascade1.detectMultiScale(gray, 1.4, 1)

    for (x, y, w, h) in cars:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        for (x, y, w, h) in cars1:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0,255), 2)

    print("number of bike- {}".format(len(cars)))
    print("number of heads- {}".format(len(cars1)))

    cv2.imshow('video', img)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()