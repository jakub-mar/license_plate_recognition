import numpy as np
import cv2 as cv

defTrackbarMin = 30
defTrackbarMax = 45


# def readLetters(path):
#     letters = {}
#     for letter in path.iterdir():
#         # print(letter, "\n\n")
#         letters[letter[8]] = cv.imread(letter, 0)

#     return letters


def getPlateLetters(plate):
    # hierarchy, contours =
    pass


def getPlateNumbers(plate, image, i):
    ret, thresh = cv.threshold(plate, 200, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    cv.drawContours(image, contours, -1, (255, 0, 0), 5)

    for i, cnt in enumerate(contours):
        x, y, w, h = cv.boundingRect(cnt)
        if h >= thresh.shape[0] / 2:
            cv.drawContours(thresh, [cnt], -1, (0, 255, 0), 5)

    cv.imshow(f"plate_{i}", image)


def getContrast(
    image: np.ndarray, topHatSize: int, blackHatSize: int, dilateSize: int
) -> np.ndarray:
    alpha = 1.1  # Contrast control (1.0-3.0)
    beta = 1  # Brightness control (0-100)

    adjusted = cv.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted


def getPlate(image: np.ndarray, min: int, max: int):
    curImg = image
    canny = cv.Canny(image, min, max)

    dilation_size = 9
    element = cv.getStructuringElement(
        cv.MORPH_RECT,
        (dilation_size, dilation_size),
    )
    dilated = cv.erode(canny, np.ones((5, 5), np.uint8))
    dilated = cv.morphologyEx(dilated, cv.MORPH_OPEN, np.ones((5, 5), np.uint8))
    dilated = cv.morphologyEx(
        dilated, cv.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1
    )
    dilated = cv.dilate(canny, element, iterations=1)

    contours, hierarchy = cv.findContours(dilated, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    plateContour = []
    candidates = []
    candidates_boxes = []
    hierarchy = hierarchy[0]
    for i, contour in enumerate(contours):
        x1, y1 = contour[0][0]
        approx = cv.approxPolyDP(contour, 0.02 * cv.arcLength(contour, True), True)
        if len(approx) == 4:
            x, y, w, h = cv.boundingRect(contour)
            ratio = float(h / w)
            area = cv.contourArea(contour)
            if (
                area >= ((image.shape[0] / 3) * (image.shape[1] / 3)) * 0.3
                and ratio >= 0.15
                and ratio <= 0.5
            ):
                plateContour.append(contour)
                brect = cv.boundingRect(approx)
                x, y, w, h = brect
                candidates_boxes.append(brect)
                candidates.append(
                    image[
                        int(y * 0.98) : int((y + h) * 1.02),
                        int(x * 0.98) : int((x + w) * 1.02),
                    ]
                )

    return candidates, candidates_boxes


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
    blurred = cv.bilateralFilter(gray, 20, 50, 50)

    min = cv.getTrackbarPos("min", winName)
    max = cv.getTrackbarPos("max", winName)
    defTrackbarMax = max
    defTrackbarMin = min
    candidates, candidates_boxes = getPlate(blurred, min, max)

    numbers = []
    for i, can in enumerate(candidates):
        # cv.imshow(f"cand_{i}", cv.resize(can, (1100, 500)))
        x, y, w, h = candidates_boxes[i]
        numbers.append(
            getPlateNumbers(
                can,
                image[
                    int(y * 0.98) : int((y + h) * 1.02),
                    int(x * 0.98) : int((x + w) * 1.02),
                ],
                i,
            )
        )

    image = cv.resize(image, (800, 600))

    if cv.waitKey(0) == ord("q"):
        cv.destroyAllWindows()
    return "PO12345"
