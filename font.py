import numpy as np
import cv2 as cv


font = cv.imread("./font.png")
font = cv.copyMakeBorder(font, 5, 5, 5, 5, cv.BORDER_CONSTANT, None, (255, 255, 255))
img = cv.cvtColor(font, cv.COLOR_BGR2GRAY)

contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
# contours = contours[1:]

hierarchy = hierarchy[0]
cv.drawContours(font, contours, -1, (0, 0, 255), 2)

contours_poly = [None] * len(contours)
boundRect = [None] * len(contours)
centers = [None] * len(contours)
radius = [None] * len(contours)
for i, c in enumerate(contours):
    if hierarchy[i][3] == 0:
        print(i, "===", hierarchy[i], "\n\n")
        contours_poly.append(cv.approxPolyDP(c, 3, True))
        # boundRect.append()

drawing = np.zeros((font.shape[0], font.shape[1], 3), dtype=np.uint8)
for i in range(len(contours_poly)):
    color = (255, 0, 0)
    # cv.drawContours(drawing, contours_poly, i, color)
    rect = cv.boundingRect(contours_poly[i])
    cv.rectangle(
        drawing,
        (int(rect[0]), int(rect[1])),
        (
            int(rect[0] + rect[2]),
            int(rect[1] + rect[3]),
        ),
        color,
        2,
    )


cv.imshow("font", font)
cv.imshow("drawing", drawing)
cv.waitKey(0)
cv.destroyAllWindows()
