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


def getPlateLetters(plate, letters):
    if not plate.shape or plate.all(None):
        return "PO13245"
    plateG = cv.cvtColor(plate, cv.COLOR_BGR2GRAY)
    # cv.imshow("plateG", plateG)
    ctr = cv.findContours(plateG, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    if not ctr or ctr[1].any(None):
        return "PO13245"
    contours2 = imutils.grab_contours(ctr)
    (contours2, bboxes) = contours.sort_contours(contours2, method="left-to-right")
    letters_candidates = []
    for i, cnt in enumerate(contours2):
        x, y, w, h = cv.boundingRect(cnt)
        approx = cv.approxPolyDP(cnt, 0.05 * cv.arcLength(cnt, True), True)
        if h >= plate.shape[0] / 3 and (w / h) >= 0.25 and (w / h) <= 1.2:
            letters_candidates.append(cnt)
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
    plate = cv.blur(plate, (7, 7))

    # threshold otsu
    ret, threshOtsu = cv.threshold(plate, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
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
    # if hierarchy != None and len(hierarchy):
    hierarchy = hierarchy[0]
    contoursToDrawOtsu = []
    contoursToDrawAdapt = []
    for i, cnt in enumerate(contoursOtsu):
        approx = cv.approxPolyDP(cnt, 0.05 * cv.arcLength(cnt, True), True)
        (x, y), (w, h), angle = cv.minAreaRect(cnt)
        if cv.contourArea(cnt) > 4000 and len(approx) == 4:
            if ((w / h) <= 6 and (w / h) >= 3) or ((w / h) >= 0.1 and (w / h) <= 0.38):
                contoursToDrawOtsu.append(cnt)

    for i, cnt in enumerate(contoursAdapt):
        approx = cv.approxPolyDP(cnt, 0.05 * cv.arcLength(cnt, True), True)
        (x, y), (w, h), angle = cv.minAreaRect(cnt)
        if cv.contourArea(cnt) > 4000 and len(approx) == 4:
            if ((w / h) <= 6 and (w / h) >= 3) or ((w / h) >= 0.1 and (w / h) <= 0.38):
                contoursToDrawAdapt.append(cnt)

    # try:
    if len(contoursToDrawOtsu) and len(contoursToDrawAdapt):
        print("in")
        contOtsu = sorted(contoursToDrawOtsu, key=cv.contourArea, reverse=True)[0]
        contAdapt = sorted(contoursToDrawAdapt, key=cv.contourArea, reverse=True)[0]
        cv.drawContours(maskOtsu, [contOtsu], -1, (255, 255, 255), -1)
        cv.drawContours(maskAdaptive, [contAdapt], -1, (255, 255, 255), -1)
        maskBoth = cv.bitwise_and(maskAdaptive, maskOtsu)
        if np.mean(maskBoth) < 0.1:
            return np.array([])
        contoursMask, hierarchyMAsk = cv.findContours(
            cv.cvtColor(maskBoth, cv.COLOR_BGR2GRAY),
            cv.RETR_EXTERNAL,
            cv.CHAIN_APPROX_SIMPLE,
        )
        # cv.imshow(
        #     f"threshmask_{candidateNum}+3+",
        #     cv.resize(
        #         cv.drawContours(maskBoth, [contoursMask[0]], -1, (0, 255, 0), 4),
        #         (520 * 2, 114 * 2),
        #     ),
        # )
        approx = cv.convexHull(contoursMask[0])
        LT = [0, 0]
        LB = [0, maskBoth.shape[0]]
        RT = [maskBoth.shape[1], 0]
        RB = [maskBoth.shape[1], maskBoth.shape[0]]
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

        res = cv.bitwise_and(threshOtsu, maskBoth)
        pts2 = np.array(
            [[0, 0], [0, 114 * 2], [520 * 2, 114 * 2], [520 * 2, 0]], np.float32
        )
        # print("approx", np.float32(approx.reshape(4, 2)), pts2, "\n\n\n\n", sep="\n\n")
        matrix = cv.getPerspectiveTransform(
            np.float32(np.array(maskEdges).reshape(4, 2)), pts2
        )
        result = cv.warpPerspective(
            cv.cvtColor(res, cv.COLOR_BGR2GRAY), matrix, (520 * 2, 114 * 2)
        )

        # resG = cv.cvtColor(res, cv.COLOR_BGR2GRAY)
        # c, h = cv.findContours(resG, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        # cv.imshow(
        #     f"threshmask_{candidateNum}+3+", cv.resize(maskBoth, (520 * 2, 114 * 2))
        # )
        # cv.imshow(f"threshmask_{candidateNum}+3+", maskBoth)
        cv.imshow(f"threshmask_{candidateNum}+3+", result)
        return res

    print("out")
    # except:
    #     return np.array([])

    # cv.imshow(f"thresh_{candidateNum}++", cv.resize(thresh2, (800, 600)))
    # print(type(res))
    # cv.imshow(f"plate_{i}", cv.resize(image, (800, 600)))
    # if not len(contoursToDrawOtsu):


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
        # numbers.append(getPlateLetters(whitePlate, letters))

    image = cv.resize(image, (800, 600))

    if cv.waitKey(0) == ord("q"):
        cv.destroyAllWindows()

    # print(numbers)
    # print(numbers.sort(key=len, reverse=True))
    return "P012345"
