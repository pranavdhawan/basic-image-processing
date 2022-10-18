import imutils
import cv2
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
cv2.imshow("Original", image)
cv2.waitKey(0)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray", gray)
cv2.waitKey(0)

# edge detection

edged = cv2.Canny(gray, 30, 150)
cv2.imshow("Edged", edged)
cv2.waitKey(0)

thresh = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY_INV)[1]
cv2.imshow("Thresh", thresh)
cv2.waitKey(0)

cnts = cv2.findContours(thresh.copy(), 
    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
output = image.copy()

for c in cnts:
    cv2.drawContours(output, [c], -1, (240, 0, 159), 3)

cv2.imshow("Contours", output)
cv2.waitKey(0)

text = "I found {} objects!".format(len(cnts))
cv2.putText(output, text, (10, 25), 
    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
    (240, 0, 159), 2)
cv2.imshow("Contours", output)
cv2.waitKey(0)