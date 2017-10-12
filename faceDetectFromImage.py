import cv2

# Get user supplied values
imagePath = "/home/evilport/PycharmProjects/FaceDetect/sample pics/3.png"
faceCascPath = "haarcascade_frontalface_default.xml"
eyeCascPath = "haarcascade_eye.xml"
smileCascPath = "haarcascade_smile.xml"

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(faceCascPath)
eyeCascade = cv2.CascadeClassifier(eyeCascPath)
smileCascade = cv2.CascadeClassifier(smileCascPath)

# Read the image
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=2.1,
    minNeighbors=5,
    minSize=(10, 10),
    flags=cv2.CASCADE_SCALE_IMAGE
)

# Draw a rectangle around the faces and eyes
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = image[y:y+h, x:x+w]
    eyes = eyeCascade.detectMultiScale(roi_gray)
    smiles = smileCascade.detectMultiScale(roi_gray)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 0, 255), 2)
    for (sx, sy, sw, sh) in smiles:
        cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (255, 0, 0), 2)

cv2.imshow("Features found", image)
cv2.waitKey(0)