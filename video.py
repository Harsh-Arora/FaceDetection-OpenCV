import cv2

video = cv2.VideoCapture(0)
cascade_face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cascade_eye = cv2.CascadeClassifier('haarcascade_eye.xml')
frames = 1
faces = 0
while True:
    check, frame = video.read()
    frames = frames + 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade_face.detectMultiScale(gray, 1.1, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, "Person", (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 0, 0), 2)
        print(x, y)
        roi = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = cascade_eye.detectMultiScale(roi, 1.1, 5)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            cv2.putText(roi_color, "Eye", (ex, ey - 5), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 2)
            print(ex, ey)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

print("Buffered {0} frames".format(frames))
print("Found {0} face(s)!".format(len(faces)))
video.release()
cv2.destroyAllWindows()