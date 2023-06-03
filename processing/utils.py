import numpy as np
import cv2 as cv


# def readLetters(path):
#     letters = {}
#     for letter in path.iterdir():
#         # print(letter, "\n\n")
#         letters[letter[8]] = cv.imread(letter, 0)

#     return letters


def getPlateLetters(plate):
    if not plate:
        return ""
    # hierarchy, contours =
    pass


def getWhitePlate(plate, image, i):
    candidateNum = i
    plate = cv.GaussianBlur(plate, (5, 5), 12)
    ret, thresh = cv.threshold(plate, 90, 255, cv.THRESH_OTSU)
    thresh = cv.erode(thresh, np.ones((4, 4), np.uint8), iterations=1)
    contours, hierarchy = cv.findContours(
        thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE
    )
    thresh2 = cv.cvtColor(thresh, cv.COLOR_GRAY2BGR)
    mask = np.zeros_like(thresh2)
    hierarchy = hierarchy[0]
    contoursToDraw = []
    for i, cnt in enumerate(contours):
        approx = cv.approxPolyDP(cnt, 0.05 * cv.arcLength(cnt, True), True)
        (x, y), (w, h), angle = cv.minAreaRect(cnt)
        if cv.contourArea(cnt) > 4000 and len(approx) == 4:
            if ((w / h) <= 6 and (w / h) >= 3) or ((w / h) >= 0.1 and (w / h) <= 0.38):
                contoursToDraw.append(cnt)

    if len(contoursToDraw):
        cont = sorted(contoursToDraw, key=cv.contourArea, reverse=True)[0]
        cv.drawContours(mask, [cont], -1, (255, 255, 255), -1)
        res = cv.bitwise_and(thresh2, mask)
        # cv.imshow(f"threshmask_{candidateNum}++", cv.resize(res, (800, 600)))
        return res
    # cv.imshow(f"thresh_{candidateNum}++", cv.resize(thresh2, (800, 600)))
    # print(type(res))
    # cv.imshow(f"plate_{i}", cv.resize(image, (800, 600)))
    if not len(contoursToDraw):
        return None


def getContrast(
    image: np.ndarray, topHatSize: int, blackHatSize: int, dilateSize: int
) -> np.ndarray:
    alpha = 1.1  # Contrast control (1.0-3.0)
    beta = 1  # Brightness control (0-100)

    adjusted = cv.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted


def getPlate(image: np.ndarray):
    curImg = image
    canny = cv.Canny(image, 30, 45)

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
                        int(y * 0.94) : int((y + h) * 1.06),
                        int(x * 0.94) : int((x + w) * 1.06),
                    ]
                )

    return candidates, candidates_boxes


def onTrack(arg):
    pass


def perform_processing(image: np.ndarray) -> str:
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray = getContrast(gray, 3, 3, 0)
    blurred = cv.bilateralFilter(gray, 20, 50, 50)

    candidates, candidates_boxes = getPlate(blurred)

    numbers = []
    for i, can in enumerate(candidates):
        # cv.imshow(f"cand_{i}", cv.resize(can, (1100, 500)))
        x, y, w, h = candidates_boxes[i]
        getWhitePlate(
            can,
            image[
                int(y * 0.98) : int((y + h) * 1.02),
                int(x * 0.98) : int((x + w) * 1.02),
            ],
            i,
        )
        # numbers.append(getPlateLetters(whitePlate))

    image = cv.resize(image, (800, 600))

    if cv.waitKey(0) == ord("q"):
        cv.destroyAllWindows()
    return "PO12345"
