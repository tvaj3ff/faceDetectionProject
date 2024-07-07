import cv2 as cv

face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

video_capture = cv.VideoCapture(0)
if not video_capture:
    print("Error: Could not open de webcam.")
    exit()

while True:
    ret, frame = video_capture.read()
    if not ret:
        print('Error: Failed to capture image.')

    gray_scale = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_scale, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # display the result
    cv.imshow('Metrix - Face Detector', frame)

    # break loop option
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv.destroyAllWindows()
