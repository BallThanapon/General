import cv2
import numpy as np
import time
import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(title='Select Video')
print(file_path)

# file_path = 'SynxIPcam_urn-uuid-643C9869-12C593A8-001D-0000-000066334873_2020-11-01_05-53-00(1).mp4'
cap = cv2.VideoCapture(file_path)
CONFIDENCE_THRESHOLD = 0.1
NMS_THRESHOLD = 0.1
weights = "../Hawk_eye/hwakeye/HAWKEYE_CRANE_DETECTION/yolov4-newdataset2.weights"
cfg = "../Hawk_eye/hwakeye/HAWKEYE_CRANE_DETECTION/yolov4-newdataset2.cfg"

#class_names = ["crane","crane_boom","crane_outrigger","person"]
class_names = ["crane","person"]

# sequence_names = ["A","D","B","person"]
# sequence_names_mean = ["CLOSED","OPEN BOOM","LEG EXTENDED","person"]

COLORS = (0, 255, 255)

net = cv2.dnn.readNetFromDarknet(cfg, weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
interval_img = 0
layer = net.getLayerNames()
layer = [layer[i[0] - 1] for i in net.getUnconnectedOutLayers()]
writer = None
Result = []

def detect(frm, net, ln):
    global Result,c,interval_count_person,interval_img, prev_count
    global end_time,start_time
    (H, W) = frm.shape[:2]
    blob = cv2.dnn.blobFromImage(frm, 1.0/255.0, (416, 416),
        swapRB=True, crop=False)
    net.setInput(blob)
    start_time = time.time()
    layerOutputs = net.forward(ln)
    end_time = time.time()

    boxes = []
    classIds = []
    confidences = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            
            if confidence > CONFIDENCE_THRESHOLD:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width/2))
                y = int(centerY - (height/2))

                boxes.append([x, y, int(width), int(height)])
                classIds.append(classID)
                confidences.append(float(confidence))
    return boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD,classIds


count_L = 0
count_S = 0
count_Crane = 0

def save_bnb(frm,boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD,classIds):
    global count_S,count_Crane,count_L
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    print("======",idxs)
    if len(idxs.flatten()) > 0 :
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            Result = np.array(frm[y:y+h,x:x+w])

            if classIds[i] == 1 :

                if Result.shape[0] > 100 :
                    try :
                        cv2.imwrite("capture/person/large/" + str(count_L) + ".jpg",Result)
                        count_L += 1
                        print("save large person")   
                    except : 
                        print ("!_img.empty()")
                else :
                    try :
                        cv2.imwrite("capture/person/small/" + str(count_S) + ".jpg",Result)
                        count_S += 1
                        print("save small person")
                    except : 
                        print ("!_img.empty()")
            else :
                try :
                    cv2.imwrite("capture/crane/" + str(count_Crane) + ".jpg",Result)
                    count_Crane += 1
                    print("save_crane")
                except :
                    print ("!_img.empty()")
    return True

def draw_bnb(frm,boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD,classIds):

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

    if len(idxs.flatten()) > 0 :
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            cv2.rectangle(frm, (x, y), (x + w, y + h), (5,255,5), 2)
            # text = "{}: {:.4f}".format(class_names[classIds[i]], confidences[i])
            text = "{}".format(class_names[classIds[i]])
            cv2.putText(frm, text, (x,y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
            fps_label = "FPS: %.2f" % (1 / (end_time - start_time))
            cv2.putText(frm, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    return frm


if __name__ == '__main__' :

    count_frm = 0
    boxes = []
    classIds = []
    confidences = []
    boxes_ = []
    confidences_ = []
    CONFIDENCE_THRESHOLD_ = []
    NMS_THRESHOLD_ = []
    classIds_ = []
    while True :
        ret,frm = cap.read()
        if not ret :
            continue

        if count_frm >= 50 :
            boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD, classIds = detect(frm,net,layer)
            if len(boxes) > 0 :
                save_bnb(frm , boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD, classIds)
            boxes_ = boxes
            confidences_ = confidences
            CONFIDENCE_THRESHOLD_ = CONFIDENCE_THRESHOLD
            NMS_THRESHOLD_ = NMS_THRESHOLD
            classIds_ = classIds
            count_frm = 0

        if len(boxes_) > 0:
            frm = draw_bnb(frm, boxes_, confidences_, CONFIDENCE_THRESHOLD_, NMS_THRESHOLD_, classIds_)

        if frm.shape[0] > 0 :
            cv2.imshow('frame',frm)
        count_frm += 1
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
