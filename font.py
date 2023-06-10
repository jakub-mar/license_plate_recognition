import numpy as np
import cv2 as cv


font = cv.imread("./font_large.png")
font = cv.copyMakeBorder(font, 5, 5, 5, 5, cv.BORDER_CONSTANT, None, (255, 255, 255))
img = cv.cvtColor(font, cv.COLOR_BGR2GRAY)

contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
# contours = contours[1:]

hierarchy = hierarchy[0]
cv.drawContours(font, contours, -1, (0, 0, 255), 1)

contours_poly = [None] * len(contours)
boundRect = [None] * len(contours)
centers = [None] * len(contours)
radius = [None] * len(contours)
for i, c in enumerate(contours):
    if hierarchy[i][3] == 0:
        # print(i, "===", hierarchy[i], "\n\n")
        contours_poly.append(cv.approxPolyDP(c, 3, True))
        # boundRect.append()

drawing = font.copy()
# drawing = np.zeros((font.shape[0], font.shape[1], 3), dtype=np.uint8)
for i in range(len(contours_poly)):
    color = (255, 0, 0)
    # cv.drawContours(drawing, contours_poly, i, color)
    x, y, w, h = cv.boundingRect(contours_poly[i])
    cv.rectangle(
        drawing,
        (int(x), int(y)),
        (
            int(x + w),
            int(y + h),
        ),
        color,
        1,
    )
    # if (x or y or w or h) == 0:
    #     break
    # roi = font[y : y + h, x : x + w]
    # print(roi.shape)
    # # cv.imshow("roi", roi)
    # while True:
    #     if cv.waitKey(30) == ord("q"):
    #         break
    # cv.imwrite(f"./letters/{i}_box.jpg", roi)
# letters = [
#     "A",
#     "B",
#     "C",
#     "D",
#     "E",
#     "F",
#     "G",
#     "H",
#     "I",
#     "J",
#     "K",
#     "L",
#     "M",
#     "N",
#     "O",
#     "P",
#     "Q",
#     "R",
#     "S",
#     "T",
#     "U",
#     "V",
#     "W",
#     "X",
#     "Y",
#     "Z",
# ]
letters = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "0",
    "t1",
    "t2",
    "t3",
    "t4",
    "t5",
    "t6",
    "t7",
    "t8",
    "t9",
    "t10",
    "t11",
    "t12",
]
for i in range(len(contours_poly)):
    rect = cv.boundingRect(contours_poly[i])
    x, y, w, h = rect
    roi = font[y : y + h, x : x + w]
    print(i, *rect, sep="   ")
    if any(v != 0 for v in rect):
        cv.imwrite(
            f"./letters2/{i}.jpg",
            cv.resize(roi, (64, 64), interpolation=cv.INTER_AREA),
        )


cv.imshow("roi", font[0:10, 0:10])
cv.imshow("font", font)
cv.imshow("drawing", drawing)
cv.waitKey(0)
cv.destroyAllWindows()
