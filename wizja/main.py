import numpy as np
import timeit
import cv2

def loadimage(nazwa):
    image = cv2.pyrDown(cv2.imread(nazwa, 0))
    wynikowy = cv2.pyrDown(cv2.imread(nazwa, 1))

#rozpoczecie zliczania czasu dzialania programu
start = timeit.default_timer()

#wczytanie pomniejszeonego obrazu w dwoch kopiach - szarej i kolorowej
image = cv2.pyrDown(cv2.imread("0.bmp",0))
wynikowy = cv2.pyrDown(cv2.imread("0.bmp",1))
#progowanie adaptacyjne (ze wzgledu na niejednorodne oswietlenie
image = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,5,0)
#operacje morfologiczne
kernel = np.ones((3,3), np.uint8)
image = cv2.dilate(image,kernel)
image = cv2.erode(image,kernel)
#znalezienie konturow
pomocnicza,cnt,hierarchy = cv2.findContours(image,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

#wybranie jedynie konturow, ktore maja obszar zawierajacy sie w pewnym przedziale wartosci
cnt = [c for c in cnt if cv2.contourArea(c) > 20000 and cv2.contourArea(c) < 30000]

#dla kazdego konturu:
for c in cnt:
    #wyznaczenie minimalnego opinajacego je rownolegloboku
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    #wyrugowanie wspolrzednych rownolegloboku
    box = np.int0(box)
    #rysowanie rownolegloboku na kolorowej kopii obrazka
    #cv2.drawContours(wynikowy,[box],0,(0,0,255))

    #znalezienie srodka ciezkosci konturu
    M = cv2.moments(c)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    #narysowanie punktu w srodku ciezkosci
    #cv2.circle(wynikowy, (cX, cY), 4, (255, 0, 0), -1)
    #narysowanie punktu w srodku rownolegloboku
    #cv2.circle(wynikowy, (int(rect[0][0]), int(rect[0][1])), 2, (255, 255, 0), -1)
    #cv2.line(wynikowy,(cX, cY),(int(rect[0][0]), int(rect[0][1])),(0,255,255))
    #znalezienie kata pomiedzy srodkiem ciezkosci a srodkiem rownolegloboku
    kat = np.arctan2(rect[0][1]-cY,rect[0][0]-cX)
    print(kat)
    stopnie = (360-(kat* (180.0 / np.pi)))%360
    """cv2.putText(wynikowy, "{}".format(stopnie), (cX - 20, cY - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)"""

    #okrag u podstawy kazdej z butelek
    podstawa=[]
    for punkty in box:
        if np.linalg.norm(punkty-(cX, cY)) < np.linalg.norm(punkty-(rect[0][0],rect[0][1])):
            podstawa.append(punkty)

    cv2.circle(wynikowy,(int(abs(podstawa[0][0] + podstawa[1][0])/2),
                         int(abs(podstawa[0][1] + podstawa[1][1])/2)),
               5,(255,0,255),-1)
    kat2 = np.arctan2(cY - (podstawa[0][1] + podstawa[1][1])/2,
                      cX - (podstawa[0][0] + podstawa[1][0])/2)
    #print(kat2)
    stopnie2 = ((kat2 * (180.0 / np.pi))-90) % 360
    cv2.putText(wynikowy, "kat: {}".format(stopnie2), (cX - 40, cY - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
    cv2.putText(wynikowy, "y: {}".format(abs(podstawa[0][1] + podstawa[1][1])/2), (cX - 40, cY - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
    cv2.putText(wynikowy, "x: {}".format(abs(podstawa[0][0] + podstawa[1][0])/2), (cX - 40, cY - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
    #print(rect[1][0], rect[1][1])
    """print(a,b)
print(np.cos(np.deg2rad((stopnie+180)%360))*b/2)
print(np.sin(np.deg2rad((stopnie+180)%360))*a/2)"""
    #cv2.circle(wynikowy,(int(rect[0][0] - np.cos(np.deg2rad((stopnie+180)%360))*b/2),
    #                     int(rect[0][1] - np.sin(np.deg2rad((stopnie+180)%360))*a/2)),5,(255,0,255))

#wyswietlenie obrazka
cv2.imshow("obraz",wynikowy)

#skonczenie zliczania czasu
stop = timeit.default_timer()
print("Czas wykonywania programu: {:,.2f} s.".format(stop - start))

#oczekiwanie na wcisniecie klawisza "escape"
cv2.waitKey(0)
cv2.destroyAllWindows()
