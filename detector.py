import cv2

cap = cv2.VideoCapture(0)  # видео поток с веб камеры

cap.set(3, 700)  # Установление длины окна
cap.set(4, 700)  # Ширина окна

motion_detect = 0  # режим обнаружения движения
md_switch = 'OFF'

ret, frame1 = cap.read()
ret, frame2 = cap.read()

while True:

    status = "NOT DETECTED"  # статус движения
    keyPress = cv2.waitKey(20)
    # нахождение разницы двух кадров, которая проявляется лишь при изменении одного из них, т.е. с этого момента
    # программа реагирует на любое движение.
    diff = cv2.absdiff(frame1, frame2)
    # преобразование в оттенки серого, поскольку алгоритм вычитания фона использует
    # черно-белые пиксельные данные
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    # фильтрация лишних контуров, шумов
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # метод для выделения кромки объекта белым цветом
    # если интенсивность пикселей превышает установленное пороговое значение,
    # значение устанавливается на 255, иначе устанавливается на 0 (черный).
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    # данный метод расширяет выделенную на предыдущем этапе область
    dilated = cv2.dilate(thresh, None, iterations=3)
    # нахождение массива контурных точек
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE,
                                   cv2.CHAIN_APPROX_SIMPLE)
    # выводится надпись с информацией включен ли детектор движениий
    cv2.putText(frame1, "Motion Detection: {}".format(md_switch), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0),
                3,
                cv2.LINE_AA)
    # Цикл по контурам для поиска изменений
    for contour in contours:
        # Если контур меньше минимальной указанной площади, он не представляет
        # значительное движение и, следовательно, его следует игнорировать
        if cv2.contourArea(contour) < 1000:
            continue

        status = "DETECTED"
        # если режим обнаружения движения включен, выделяеются красные контуры при движении
        if motion_detect == 1:
            cv2.drawContours(frame1, contours, -1, (0, 0, 255), 2)
    # изменение цвета шрифта: красный- при движении, иначе зеленый
    font_color = (0, 255, 0) if status == 'NOT DETECTED' else (0, 0, 255)
    if motion_detect == 1:
        cv2.putText(frame1, "Status: {}".format(status), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, font_color, 3,
                    cv2.LINE_AA)
    cv2.imshow("frame1", frame1)
    frame1 = frame2
    ret, frame2 = cap.read()

    if cv2.waitKey(20) == 27:
        break

    if keyPress & 0xFF == ord('s'):
        motion_detect = 0
        md_switch = 'OFF'

    if keyPress & 0xFF == ord('m'):
        motion_detect = 1
        md_switch = 'ON'

cap.release()
cv2.destroyAllWindows()
