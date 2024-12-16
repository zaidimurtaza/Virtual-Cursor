import mediapipe as mp
import cv2
import time

class HandGestures(): 
    def __init__(self,
               mode=False,
               max_hands=2,
               model_complexity=1,
               detection_confidence=0.5,
               tracking_confidence=0.5):
        self.mode = mode
        self.max_hand = max_hands
        self.model_complexity = model_complexity
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence
        
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.mode,self.max_hand,self.model_complexity,self.detection_confidence,self.tracking_confidence)
        self.mp_draw = mp.solutions.drawing_utils
    
    def findHands(self,img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)
        # print(results.multi_hand_landmarks)
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        return img
    
    def findPosition(self, img, hand_no = 0, draw = True):
        land_mark_list = []
        if self.results.multi_hand_landmarks:
            # try:
            my_hand = self.results.multi_hand_landmarks[hand_no]
            # except:
            #     my_hand = self.results.multi_hand_landmarks[0]
            for id, lm in enumerate(my_hand.landmark):
                h,w,c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                land_mark_list.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx,cy), 7, (255,0,0), cv2.FILLED)
        return land_mark_list

def main():
    capture = cv2.VideoCapture(0)
    current_tm = 0
    past_tm = 0
    detector = HandGestures()
    while True:
        sucess,img = capture.read()
        img1 = detector.findHands(img,False)
        point_on_finger = detector.findPosition(img)
        if len(point_on_finger) != 0:
            print(point_on_finger[4],point_on_finger[8])
            
        current_tm = time.time()
        frame_rate = 1/(current_tm - past_tm)
        past_tm = time.time() 
                
        cv2.putText(img,str(int(frame_rate)),(10,70),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,255),3)
        
        cv2.imshow("Image",img)
        cv2.waitKey(1)

if __name__== "__main__":
    main()