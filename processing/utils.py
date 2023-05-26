import numpy as np
import cv2 as cv

defTrackbarMin = 30
defTrackbarMax = 65


def getContrast(
    image: np.ndarray, topHatSize: int, blackHatSize: int, dilateSize: int
) -> np.ndarray:
    # kernelTop = cv.getStructuringElement(cv.MORPH_ELLIPSE, (topHatSize, topHatSize))
    # kernelBlack = cv.getStructuringElement(
    #     cv.MORPH_ELLIPSE, (blackHatSize, blackHatSize)
    # )
    # tophat = cv.morphologyEx(image, cv.MORPH_TOPHAT, kernelTop)
    # blackhat = cv.morphologyEx(image, cv.MORPH_BLACKHAT, kernelBlack)

    # addition = cv.add(tophat, image)
    # subtraction = cv.subtract(addition, blackhat)

    alpha = 1.1  # Contrast control (1.0-3.0)
    beta = 1  # Brightness control (0-100)

    adjusted = cv.convertScaleAbs(image, alpha=alpha, beta=beta)
    # kerneldilate = cv.getStructuringElement(cv.MORPH_RECT, (dilateSize, dilateSize))
    # result = cv.dilate(subtraction, kerneldilate)
    # cv.imshow("subtraction", cv.resize(adjusted, (800, 600)))
    return adjusted


def getPlate(image: np.ndarray, min: int, max: int):
    curImg = image
    canny = cv.Canny(image, min, max)

    dilation_size = 5
    # (2 * dilation_size + 1, 2 * dilation_size + 1),
    #     (dilation_size, dilation_size),
    element = cv.getStructuringElement(
        cv.MORPH_RECT,
        (dilation_size, dilation_size),
        # (dilation_size, dilation_size),
    )
    # np.ones((7, 7), np.uint8)
    dilated = cv.dilate(canny, element, iterations=1)

    contours, hierarchy = cv.findContours(dilated, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    plateContour = []
    for contour in contours:
        x1, y1 = contour[0][0]
        approx = cv.approxPolyDP(contour, 0.011 * cv.arcLength(contour, True), True)
        if len(approx) == 4:
            x, y, w, h = cv.boundingRect(contour)
            ratio = float(h / w)
            area = cv.contourArea(contour)
            if ratio >= 0.15 and ratio <= 0.65 and area >= 90000:
                plateContour.append(contour)
    return dilated, plateContour, contours


def onTrack(arg):
    pass


def perform_processing(image: np.ndarray) -> str:
    global defTrackbarMax, defTrackbarMin
    winName = "trackbars"
    cv.namedWindow(winName)
    cv.createTrackbar("min", winName, defTrackbarMin, 255, onTrack)
    cv.createTrackbar("max", winName, defTrackbarMax, 255, onTrack)

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray = getContrast(gray, 3, 3, 0)
    blurred = cv.GaussianBlur(gray, (7, 7), 0)
    blurred = cv.morphologyEx(blurred, cv.MORPH_OPEN, np.ones((5, 5), np.uint8))

    while True:
        min = cv.getTrackbarPos("min", winName)
        max = cv.getTrackbarPos("max", winName)
        defTrackbarMax = max
        defTrackbarMin = min
        canny, contour, contours = getPlate(blurred, min, max)
        if len(contour) != 0:
            cv.drawContours(image, contour, -1, (0, 0, 255), 10)

        canny = cv.resize(canny, (800, 600))
        image = cv.resize(image, (800, 600))

        grey_3_channel = cv.cvtColor(canny, cv.COLOR_GRAY2BGR)

        numpy_horizontal = np.hstack((image, grey_3_channel))

        numpy_horizontal_concat = np.concatenate((image, grey_3_channel), axis=1)
        cv.imshow(winName, numpy_horizontal_concat)
        # cv.imshow("gray", cv.resize(gray, (800, 600)))

        if cv.waitKey(30) == ord("q"):
            cv.destroyAllWindows()
            break
    return "PO12345"
