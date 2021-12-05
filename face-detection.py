import cv2
# Load the cascade 
pathf = 'path\\to\\haarcascades\\xml\\file'
face_cascade = cv2.CascadeClassifier(pathf)

# Read the input image
img = cv2.imread('input_photo_image.jpg')
gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# Detect faces
faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.05,minNeighbors=5)
# Draw rectangle around the faces
for (x, y, w, h) in faces: 
  cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

# Export the result
cv2.imwrite("face_detected.png", img) 
print('Successfully saved')

