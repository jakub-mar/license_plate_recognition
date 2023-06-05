import numpy as np
import cv2 as cv
import imutils
from imutils import contours


def readLetters(path):
    letters = {}
    for letter in path.iterdir():
        letterName = str(letter)[8]
        letters[letterName] = cv.imread(str(letter), 0)

    return letters


def matchLetter(letter, letters):
    results = {}
    maxResult = 0
    bestVal = None
    for l in letters:
        # print(l)
        # letters[l] = cv.cvtColor(letters[l], cv.COLOR_BGR2GRAY)
        # letter = cv.cvtColor(letter, cv.COLOR_BGR2GRAY)
        letters[l] = letters[l].astype(np.uint8)
        letter = letter.astype(np.uint8)
        # print(letters[l].shape)
        # print(letter.shape)
        # print("\n\n")
        result = cv.matchTemplate(letter, letters[l], cv.TM_CCOEFF)
        _, max_val, _, max_loc = cv.minMaxLoc(result)
        if max_val > maxResult:
            bestVal = l
            maxResult = max_val

    # print(bestVal)
    return str(bestVal)


def crop_minAreaRect(img, rect, i):
    # rotate img
    angle = rect[2]
    rows, cols = img.shape[0], img.shape[1]
    M = cv.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    img_rot = cv.warpAffine(img, M, (cols, rows))

    # # rotate bounding box
    rect0 = (rect[0], rect[1], 0.0)
    box = cv.boxPoints(rect0)
    pts = np.int0(cv.transform(np.array([box]), M))[0]
    pts[pts < 0] = 0

    # crop
    img_crop = img_rot[pts[1][1] : pts[0][1], pts[1][0] : pts[2][0]]

    # print(img_crop, "\n\n")
    # if not img_crop.all(None) or img_crop.shape:
    #     cv.imshow(f"{i}", img_crop)
    # return img_crop


def getPlateLetters(plate, letters):
    if plate.all(None) or not plate.shape:
        return ""
    plateG = cv.cvtColor(plate, cv.COLOR_BGR2GRAY)
    ctr = cv.findContours(plateG, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # cv.drawContours(plate, contours, 2, (255, 0, 0), 5)
    # cv.imshow("test", plate)
    contours2 = imutils.grab_contours(ctr)
    (contours2, bboxes) = contours.sort_contours(contours2, method="left-to-right")
    # print("contours2", contours2)
    letters_candidates = []
    for i, cnt in enumerate(contours2):
        x, y, w, h = cv.boundingRect(cnt)
        approx = cv.approxPolyDP(cnt, 0.05 * cv.arcLength(cnt, True), True)
        if h >= plate.shape[0] / 3 and (w / h) >= 0.25 and (w / h) <= 1.2:
            # cv.drawContours(plate, [cnt], -1, (0, 0, 255), 5)
            letters_candidates.append(cnt)
            # crop_minAreaRect(plate, rect, i)
    letters_sorted = sorted(
        letters_candidates,
        key=lambda a: cv.boundingRect(a)[3],
        reverse=False,
    )[:7]
    letters_roi = []
    for let in letters_candidates:
        x, y, w, h = cv.boundingRect(let)
        pts1 = np.float32([[x, y], [x + w, y], [x, y + h], [x + w, y + h]])
        pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        matrix = cv.getPerspectiveTransform(pts1, pts2)
        result = cv.warpPerspective(plateG, matrix, (w, h))
        letters_roi.append(result)
        # cv.imshow(f"test_{i}", result)
        rect = cv.minAreaRect(let)
        box = cv.boxPoints(rect)
        box = np.int0(box)

    plateString = []
    for i, cnt in enumerate(letters_roi):
        letter = cv.resize(cnt, (64, 64), interpolation=cv.INTER_AREA)
        plateString.append(matchLetter(letter, letters))
        # cv.drawContours(plate, [cnt], -1, (0, 255, 0), 3)
        # cv.imshow(f"plate_{i}", cnt)
    print("".join(plateString))
    return "".join(plateString)


def getWhitePlate(plate, image, i):
    candidateNum = i
    # plate = cv.GaussianBlur(plate, (9, 9), 7)
    plate = cv.bilateralFilter(plate, 5, 20, 20)

    # plate = cv.medianBlur(plate, 7)
    # ret, thresh = cv.threshold(plate, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    # thresh = cv.adaptiveThreshold(
    #     plate, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 1.2
    # )
    thresh = cv.adaptiveThreshold(
        plate, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 19, 0.8
    )
    thresh = cv.erode(thresh, np.ones((5, 5), np.uint8))
    # thresh = cv.dilate(thresh, (21, 21))
    # thresh = cv.morphologyEx(thresh, cv.MORPH_OPEN, (3, 3))
    thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, (3, 3))
    # thresh = cv.dilate(thresh, (1, 1))
    cv.imshow(f"raw_thresh_{i}", cv.resize(thresh, (1024, 768)))
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    thresh2 = cv.cvtColor(thresh, cv.COLOR_GRAY2BGR)
    # cv.drawContours(thresh2, contours, -1, (0, 255, 0), 4)
    # cv.imshow(f"adaptive{i}", cv.resize(thresh2, (800, 600)))
    # cv.imshow(f"test_{i}", cv.resize(thresh2, (800, 600)))
    mask = np.zeros_like(thresh2)
    # if hierarchy != None and len(hierarchy):
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
        # res = res.astype(np.uint8)
        resG = cv.cvtColor(res, cv.COLOR_BGR2GRAY)
        c, h = cv.findContours(resG, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        if len(c):
            # cv.drawContours(res, c, -1,
            # (255, 0, 0), 5)
            for cnt in c:
                # rect = cv.minAreaRect(cnt)
                rect = cv.convexHull(cnt)
                # box = cv.boxPoints(rect)
                # box = np.int0(box)
                cv.drawContours(res, [rect], -1, (255, 0, 0), 5)
        cv.imshow(f"threshmask_{candidateNum}+3+", cv.resize(res, (520 * 2, 114 * 2)))
        return res
    # cv.imshow(f"thresh_{candidateNum}++", cv.resize(thresh2, (800, 600)))
    # print(type(res))
    # cv.imshow(f"plate_{i}", cv.resize(image, (800, 600)))
    if not len(contoursToDraw):
        return np.array(None)


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


def perform_processing(image: np.ndarray, letters) -> str:
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray = getContrast(gray, 3, 3, 0)
    blurred = cv.bilateralFilter(gray, 20, 50, 50)
    # cv.imshow("orig", cv.resize(gray, (800, 600)))

    candidates, candidates_boxes = getPlate(blurred)

    numbers = []
    for i, can in enumerate(candidates):
        # cv.imshow(f"cand_{i}", cv.resize(can, (1100, 500)))
        x, y, w, h = candidates_boxes[i]
        whitePlate = getWhitePlate(
            can,
            image[
                int(y * 0.98) : int((y + h) * 1.02),
                int(x * 0.98) : int((x + w) * 1.02),
            ],
            i,
        )
        numbers.append(getPlateLetters(whitePlate, letters))

    image = cv.resize(image, (800, 600))

    if cv.waitKey(0) == ord("q"):
        cv.destroyAllWindows()

    print(numbers)
    print(numbers.sort(key=len, reverse=True))
    return numbers[0]
