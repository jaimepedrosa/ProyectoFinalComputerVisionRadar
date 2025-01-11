
import cv2
import time
import os
import numpy as np
import threading
from rastreador import *

carI = {}

tracker = Rastreador()
deteccion = cv2.createBackgroundSubtractorMOG2(history=10000,varThreshold=100)

i = 0
num_frame_actual = 0


cap = cv2.VideoCapture("../assets/video_coche.mov")

while True:
    ret, frame = cap.read()

    height = frame.shape[0]
    weight = frame.shape[1]

    mask = np.zeros((height,weight),dtype=np.uint8)

    ptos = np.array([[140,195],[198,195],[204,430],[21,430]])

    cv2.fillPoly(mask,[ptos],255)

    zona = cv2.bitwise_and(frame,frame,mask=mask)

    areag = np.array([[140,195],[198,195],[204,430],[21,430]])
    area1 = np.array([[74,320],[198,320],[204,430],[21,430]])
    area2 = np.array([[104,265],[199,265],[198,320],[74,320]])
    area3 = np.array([[140,195],[198,195],[199,265],[104,265]])

    cv2.polylines(frame, [np.array(areag,np.int32)], True, (255,255,0), 2)
    cv2.polylines(frame, [np.array(area3,np.int32)], True, (0,130,255), 1)
    cv2.polylines(frame, [np.array(area2,np.int32)], True, (0,0,255), 1)
    cv2.polylines(frame, [np.array(area2,np.int32)], True, (0,130,255), 1)

    mascara = deteccion.apply(zona)

    filtro = cv2.GaussianBlur(mascara,(11,11),0)

    _,umbral = cv2.threshold(filtro,50,255,cv2.THRESH_BINARY)

    dila = cv2.dilate(umbral, np.ones((3, 3), dtype=np.uint8))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

    cerrar = cv2.morphologyEx(dila,cv2.MORPH_CLOSE,kernel)

    contornos, _ = cv2.findContours(cerrar,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    detecciones = []

    for cont in contornos:
        area = cv2.contourArea(cont)
        if area >1000:
            x,y,ancho,alto = cv2.boundingRect(cont)

            detecciones.append([x,y,ancho,alto])
    
    info_id = tracker.rastreo(detecciones)

    for inf in info_id:
        x,y,ancho,alto, id = inf

        cv2.rectangle(frame, (x,y-10),(x+ancho,y+alto),(0,0,255),2)

        cx = int(x+ancho /2)
        cy = int(y+alto /2)

        a2 = cv2.pointPolygonTest(np.array(area2,np.int32),(cx,cy),False)

        if a2 >= 0 and id not in carI:
            carI[id] = num_frame_actual

        if id in carI:
            cv2.circle(frame, (cx,cy),3,(0,0,255),-1)

            a3 = cv2.pointPolygonTest(np.array(area3,np.int32),(cx,cy),False)

            if a3 >= 0:
                i += 1
                n_frames = num_frame_actual-carI[id]
                tiempo = n_frames/30
                print(f"Tiempo transcurrido: {tiempo}")

                vel = (0.15/tiempo)*36
                print(f"ID: {id}, Velocidad: {vel} km/h")
                #del carI[id]

                cv2.rectangle(frame,(x,y-10),(x+100,y-50),(0,0,255),-1)
                cv2.putText(frame,str(int(vel)) + " KM/H", (x,y-35), cv2.FONT_HERSHEY_PLAIN,1,(255,255,255),2)

    cv2.imshow("Video",frame)
    num_frame_actual += 1
    #print(i)
    if cv2.waitKey(100//30) == ord('q'):
        break

cv2.destroyWindow("Video")

