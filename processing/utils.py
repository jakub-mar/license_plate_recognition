import numpy as np
import cv2 as cv
import imutils
from imutils import contours
from skimage.metrics import structural_similarity as ssim


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
    values = []
    for l in letters:
        letters[l] = letters[l].astype(np.uint8)
        letter = letter.astype(np.uint8)
        result = cv.matchTemplate(letter, letters[l], cv.TM_CCOEFF)
        _, max_val, _, max_loc = cv.minMaxLoc(result)
        if max_val > maxResult:
            bestVal = l
            maxResult = max_val
            values.append(bestVal)
    return str(bestVal)


def getPlateLetters(plate, letters, i):
    if not plate.any():
        return "PO33344"
    plate3 = cv.cvtColor(plate, cv.COLOR_GRAY2BGR)
    ctr = cv.findContours(
        np.bitwise_not(plate), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
    )
    contours2 = imutils.grab_contours(ctr)
    try:
        (contours2, bboxes) = contours.sort_contours(contours2, method="left-to-right")
    except:
        pass
    letters_candidates = []
    for i, cnt in enumerate(contours2):
        x, y, w, h = cv.boundingRect(cnt)
        if h >= plate.shape[0] / 3 and (w / h) >= 0.15 and (w / h) <= 1.3:
            letters_candidates.append(cnt)
        letters_sorted = sorted(
            letters_candidates,
            key=lambda a: cv.boundingRect(a)[3],
            reverse=False,
        )
    cv.drawContours(plate3, letters_candidates, -1, (0, 255, 0), 5)
    letters_roi = []
    for let in letters_candidates:
        x, y, w, h = cv.boundingRect(let)
        if h < 0.5 * plate.shape[0]:
            continue
        pts1 = np.float32([[x, y], [x + w, y], [x, y + h], [x + w, y + h]])
        pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        matrix = cv.getPerspectiveTransform(pts1, pts2)
        result = cv.warpPerspective(plate, matrix, (w, h))

        letters_roi.append(result)
        rect = cv.minAreaRect(let)
        box = cv.boxPoints(rect)
        box = np.int0(box)
    plateString = []
    for i, cnt in enumerate(letters_roi):
        letter = cv.resize(cnt, (64, 64), interpolation=cv.INTER_AREA)
        if cv.countNonZero(letter) / (letter.shape[0] * letter.shape[1]) > 0.85:
            continue
        plateString.append(matchLetter(letter, letters))

    for i, letter in plateString:
        letter = "O" if (letter == "0" and i < 3) else letter
    return "".join(plateString)


def getWhitePlate(plate, image, i):
    candidateNum = i
    plate = cv.bilateralFilter(plate, 20, 50, 50)
    plate = cv.blur(plate, (7, 7))

    # threshold otsu
    ret, threshOtsu = cv.threshold(plate, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    threshOtsu = cv.erode(threshOtsu, np.ones((7, 7), np.uint8))
    contoursOtsu, hierarchy = cv.findContours(
        threshOtsu, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE
    )
    # threshold adaptive
    threshAdaptive = cv.adaptiveThreshold(
        plate, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 1
    )
    threshAdaptive = cv.morphologyEx(threshAdaptive, cv.MORPH_CLOSE, (7, 7))
    threshAdaptive = cv.erode(threshAdaptive, np.ones((7, 7), np.uint8))
    threshAdaptive = cv.morphologyEx(threshAdaptive, cv.MORPH_OPEN, (7, 7))
    contoursAdapt, hierarchy = cv.findContours(
        threshAdaptive, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE
    )

    # threshold to grayscale
    threshOtsu = cv.cvtColor(threshOtsu, cv.COLOR_GRAY2BGR)
    threshAdaptive = cv.cvtColor(threshAdaptive, cv.COLOR_GRAY2BGR)
    maskOtsu = np.zeros_like(threshOtsu)
    maskAdaptive = np.zeros_like(threshOtsu)
    hierarchy = hierarchy[0]
    contoursToDrawOtsu = []
    contoursToDrawAdapt = []
    mask = np.zeros_like(threshOtsu)
    for i, cnt in enumerate(contoursOtsu):
        approx = cv.approxPolyDP(cnt, 0.05 * cv.arcLength(cnt, True), True)
        (x, y), (w, h), angle = cv.minAreaRect(cnt)
        if cv.contourArea(cnt) > 4000 and len(approx) == 4:
            if ((w / h) <= 7 and (w / h) >= 2) or ((w / h) >= 0.1 and (w / h) <= 0.5):
                contoursToDrawOtsu.append(cnt)

    for i, cnt in enumerate(contoursAdapt):
        approx = cv.approxPolyDP(cnt, 0.05 * cv.arcLength(cnt, True), True)
        (x, y), (w, h), angle = cv.minAreaRect(cnt)
        if cv.contourArea(cnt) > 4000 and len(approx) == 4:
            if ((w / h) <= 7 and (w / h) >= 2) or ((w / h) >= 0.1 and (w / h) <= 0.5):
                contoursToDrawAdapt.append(cnt)
    approx = []
    meanTooLow = False
    if len(contoursToDrawOtsu) and len(contoursToDrawAdapt):
        contOtsu = sorted(contoursToDrawOtsu, key=cv.contourArea, reverse=True)[0]
        contAdapt = sorted(contoursToDrawAdapt, key=cv.contourArea, reverse=True)[0]
        cv.drawContours(maskOtsu, [contOtsu], -1, (255, 255, 255), -1)
        cv.drawContours(maskAdaptive, [contAdapt], -1, (255, 255, 255), -1)
        maskBoth = cv.bitwise_and(maskAdaptive, maskOtsu)
        if (
            cv.countNonZero(cv.cvtColor(maskBoth, cv.COLOR_BGR2GRAY))
            / (plate.shape[0] * plate.shape[1])
            > 0.3
        ):
            contoursMask, hierarchyMAsk = cv.findContours(
                cv.cvtColor(maskBoth, cv.COLOR_BGR2GRAY),
                cv.RETR_EXTERNAL,
                cv.CHAIN_APPROX_SIMPLE,
            )
            contoursMask = sorted(
                contoursMask, key=lambda x: cv.contourArea(x), reverse=True
            )
            approx = cv.convexHull(contoursMask[0])
            mask = maskBoth
        else:
            meanTooLow = True
            maskAdaptive = np.zeros_like(threshOtsu)

    if meanTooLow or not len(contoursToDrawAdapt) and len(contoursToDrawOtsu):
        contOtsu = sorted(contoursToDrawOtsu, key=cv.contourArea, reverse=True)[0]
        cv.drawContours(maskOtsu, [contOtsu], -1, (255, 255, 255), -1)
        contoursMask, hierarchyMAsk = cv.findContours(
            cv.cvtColor(maskOtsu, cv.COLOR_BGR2GRAY),
            cv.RETR_EXTERNAL,
            cv.CHAIN_APPROX_SIMPLE,
        )
        contoursMask = sorted(
            contoursMask, key=lambda x: cv.contourArea(x), reverse=True
        )
        approx = cv.convexHull(contoursMask[0])
        mask = maskOtsu

    if len(contoursToDrawAdapt) and not len(contoursToDrawOtsu):
        contAdapt = sorted(contoursToDrawAdapt, key=cv.contourArea, reverse=True)[0]
        cv.drawContours(maskAdaptive, [contAdapt], -1, (255, 255, 255), -1)
        contoursMask, hierarchyMAsk = cv.findContours(
            cv.cvtColor(maskAdaptive, cv.COLOR_BGR2GRAY),
            cv.RETR_EXTERNAL,
            cv.CHAIN_APPROX_SIMPLE,
        )
        contoursMask = sorted(
            contoursMask, key=lambda x: cv.contourArea(x), reverse=True
        )
        cv.drawContours(maskAdaptive, [contoursMask[0]], -1, (0.255, 0), 5)
        approx = cv.convexHull(contoursMask[0])
        mask = maskAdaptive

    if not mask.any():
        mask = np.zeros_like(threshOtsu)
    try:
        LT = [0, 0]
        LB = [0, plate.shape[0]]
        RT = [plate.shape[1], 0]
        RB = [plate.shape[1], plate.shape[0]]
        points = [LT, LB, RB, RT]
        maskEdges = []
        for point in points:
            distances = np.linalg.norm(
                approx.reshape(len(approx), -1) - np.array(point), axis=1
            )
            min_index = np.argmin(distances)
            maskEdges.append(
                [
                    approx.reshape(len(approx), -1)[min_index][0],
                    approx.reshape(len(approx), -1)[min_index][1],
                ]
            )

        res = cv.bitwise_and(threshOtsu, mask)
        pts2 = np.array(
            [[0, 0], [0, 114 * 2], [520 * 2, 114 * 2], [520 * 2, 0]], np.float32
        )
        matrix = cv.getPerspectiveTransform(
            np.float32(np.array(maskEdges).reshape(4, 2)), pts2
        )
        result = cv.warpPerspective(
            cv.cvtColor(res, cv.COLOR_BGR2GRAY), matrix, (520 * 2, 114 * 2)
        )
        result = cv.copyMakeBorder(result, 5, 5, 5, 5, cv.BORDER_CONSTANT, None, 255)

        if cv.countNonZero(result) / (result.shape[0] * result.shape[1]) > 0.2:
            return result
        else:
            return cv.cvtColor(threshOtsu, cv.COLOR_BGR2GRAY)
    except:
        return cv.cvtColor(threshOtsu, cv.COLOR_BGR2GRAY)


def getContrast(
    image: np.ndarray, topHatSize: int, blackHatSize: int, dilateSize: int
) -> np.ndarray:
    alpha = 1.1  # Contrast control (1.0-3.0)
    beta = 1  # Brightness control (0-100)

    adjusted = cv.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted


def getPlate(image: np.ndarray, gray):
    curImg = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    canny = cv.Canny(image, 30, 45)

    dilation_size = 5
    element = cv.getStructuringElement(
        cv.MORPH_RECT,
        (dilation_size, dilation_size),
    )
    dilated = cv.erode(canny, np.ones((7, 7), np.uint8))
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
    cv.drawContours(curImg, contours, -1, (255, 0, 0), 3)
    for i, contour in enumerate(contours):
        x1, y1 = contour[0][0]
        approx = cv.approxPolyDP(contour, 0.04 * cv.arcLength(contour, True), True)
        if len(approx) == 4:
            x, y, w, h = cv.boundingRect(contour)
            ratio = float(h / w)
            area = cv.contourArea(contour)
            if (
                area >= ((image.shape[0] / 3) * (image.shape[1] / 3)) * 0.3
                and ratio >= 0.15
                and ratio <= 0.5
            ):
                cv.drawContours(curImg, [contour], -1, (0, 255, 0), 3)
                plateContour.append(contour)
                brect = cv.boundingRect(approx)
                x, y, w, h = brect
                candidates_boxes.append(brect)
                candidates.append(
                    gray[
                        int(y * 0.94) : int((y + h) * 1.06),
                        int(x * 0.94) : int((x + w) * 1.06),
                    ]
                )
    if not len(candidates):
        threshAdaptive = cv.adaptiveThreshold(
            image, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 23, 1
        )
        threshAdaptive = cv.morphologyEx(threshAdaptive, cv.MORPH_CLOSE, (7, 7))
        threshAdaptive = cv.morphologyEx(threshAdaptive, cv.MORPH_OPEN, (7, 7))
        thColor = cv.cvtColor(threshAdaptive, cv.COLOR_GRAY2BGR)
        cntrs, h = cv.findContours(threshAdaptive, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        try:
            for i, cnt in enumerate(cntrs):
                approx = cv.approxPolyDP(cnt, 0.02 * cv.arcLength(cnt, True), True)
                (x, y), (w, h), angle = cv.minAreaRect(cnt)
                if cv.contourArea(cnt) > 8000 and len(approx) == 4:
                    cv.drawContours(thColor, [cnt], -1, (0, 255, 0), 7)
                    plateContour.append(contour)
                    brect = cv.boundingRect(approx)
                    x, y, w, h = brect
                    candidates_boxes.append(brect)
                    candidates.append(
                        gray[
                            int(y * 0.94) : int((y + h) * 1.06),
                            int(x * 0.94) : int((x + w) * 1.06),
                        ]
                    )
        except:
            pass
    return candidates, candidates_boxes


def onTrack(arg):
    pass


def perform_processing(image: np.ndarray, letters) -> str:
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    grayC = getContrast(gray, 3, 3, 0)
    blurred = cv.bilateralFilter(grayC, 20, 50, 50)

    candidates, candidates_boxes = getPlate(blurred, gray)

    numbers = []
    for i, can in enumerate(candidates):
        x, y, w, h = candidates_boxes[i]
        whitePlate = getWhitePlate(
            can,
            image[
                int(y * 0.98) : int((y + h) * 1.02),
                int(x * 0.98) : int((x + w) * 1.02),
            ],
            i,
        )
        numbers.append(getPlateLetters(whitePlate, letters, i))

    image = cv.resize(image, (800, 600))

    numbers = [num for num in numbers if len(num) <= 8]
    resNumber = numbers[0] if len(numbers) and len(numbers[0]) else "P012345"

    return resNumber
