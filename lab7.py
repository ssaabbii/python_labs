import numpy as np
import cv2
from matplotlib import pyplot as plt


def func(img1, img2):
    """
           This function uses the feature matching and findHomography
           to find known objects in a complex image.

           Parameters
           ----------
           img1: image
              a template (ghost)
           img2: image
              original image

           Returns
           -------
           img3 : image
               where ghost is marked in white color in original image:

    """
    # установливаем условие, чтобы для поиска объекта было не менее 10 совпадений
    MIN_MATCH_COUNT = 10

    # Инициализация SIFT детектора
    sift = cv2.SIFT_create()

    # поиск ключевых точек и дескрипторов с помощью SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # FLANN параметры
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    # в good хранятся все хорошие совпадения в соответствии с тестом Лоу на соотношение.
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    if len(good) > MIN_MATCH_COUNT:
        # Если найдено достаточное количество совпадений, извлекаются местоположения совпадающих ключевых точек
        # на обоих изображениях. Они передаются, чтобы найти перспективное преобразование
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        # передается набор точек с обоих изображений, cv2.findHomography найдет перспективное преобразование этого
        # объекта.
        # cv2.findhomography представляет собой матрицу 3x3,
        # которая отображает точки в одной точке в соответствующую точку на другом изображении.
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        # Как только получена эта матрица преобразования 3x3, используем ее для преобразования углов шаблона в
        # соответствующие точки на исходном изображении
        h, w = img1.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        # испьзуем cv2.perspectiveTransform(), чтобы найти объект.
        # Чтобы найти трансформацию, нужно по крайней мере четыре правильные точки.
        dst = cv2.perspectiveTransform(pts, M)

        img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)

    img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)

    plt.imshow(img3, 'gray'), plt.show()


def main():

    img1 = cv2.imread('candy_ghost.png', 0)
    img2 = cv2.imread('lab7.png', 0)
    func(img1, img2)


if __name__ == '__main__':
    main()
