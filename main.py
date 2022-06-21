import time
import cv2
import mediapipe as mp
from google.protobuf.json_format import MessageToDict

obraz = cv2.VideoCapture(0)                                                                                 #tworzymy video z kamerki
rozdzielczosc1 = 1920
rozdzielczosc2 = 1080
obraz.set(cv2.CAP_PROP_FRAME_WIDTH, rozdzielczosc1)
obraz.set(cv2.CAP_PROP_FRAME_HEIGHT, rozdzielczosc2)
obraz.set(cv2.CAP_PROP_FPS, 60)                                                                                     # ustawiamy fps
reka_klasa = mp.solutions.hands                                                                                    # ustawiamy klase tworzymy reke
reka = reka_klasa.Hands()                                                                                            # (do przetwarzania obrazu reki)
mpDraw = mp.solutions.drawing_utils                                                                                                # rysujemy złącza i połączenia
palce = [(8, 6), (12, 10), (16, 14), (20, 18)]                                                                                       # koordynaty paluchów
kciuk = (4, 2)                                                                                                                # koordynaty kciuczka
while True:
    dummy, obraz_process = obraz.read()                                                                            #dummy - wartość true false, obraz_process nasz obraz z kamery
    results = reka.process(obraz_process)                                                                          #tworzy detekcje
    multiLandMarks = results.multi_hand_landmarks

    if multiLandMarks:
        handList = []
        whichHand = results.multi_handedness[0].classification[0].label
        left = 'Left'
        right = 'Right'
        for handLms in multiLandMarks:
            mpDraw.draw_landmarks(obraz_process, handLms, reka_klasa.HAND_CONNECTIONS)
            for iteracja, wartoscZmiennej in enumerate(handLms.landmark):                                           #enumerate zwraca 2 wartości, 1-wartość iteracji, 2-wartość zmiennej
                h, w, c = obraz_process.shape                                                                    #zwraca wartości obrazu (kolumny itp)
                cx = int(wartoscZmiennej.x * w)
                cy = int(wartoscZmiennej.y * h)
                handList.append((cx, cy))                                                                      # dodaje elementy do listy
        for point in handList:
            cv2.circle(obraz_process, point, 8, (25, 255, 0), 1)                                              # ustawienie koloru
            upCount = 0
            for koordynaty in palce:
                if handList[koordynaty[0]][1] < handList[koordynaty[1]][1]:
                    upCount += 1
            if handList[kciuk[0]][0] > handList[kciuk[1]][0]:
                if whichHand == left:
                    upCount += 1
            if handList[kciuk[1]][0] > handList[kciuk[0]][0]:
                if whichHand == right:
                    upCount += 1
            cv2.putText(obraz_process, str(upCount), (150, 150), cv2.FONT_HERSHEY_PLAIN, 12, (0, 255, 0), 12)
            cv2.imshow("Counting number of fingers", obraz_process)
            cv2.waitKey(1)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
obraz.release()
cv2.destroyAllWindows()
