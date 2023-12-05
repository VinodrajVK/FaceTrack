import threading
import cv2
from deepface import DeepFace
from datetime import datetime


class FaceRecognition:
    def __init__(self):
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.counter = 0
        self.face_match = False
        self.ref_img = cv2.imread('images/Passport Photo.jpg')
        self.lock = threading.Lock()

    def match_face(self, frame):
        try:
            with self.lock:
                if DeepFace.verify(frame, self.ref_img.copy())['verified']:
                    self.face_match = True
                else:
                    self.face_match = False
        except ValueError:
            pass

    def run(self):
        while True:
            ret, frame = self.cap.read()

            if ret:
                if self.counter % 30 == 0:
                    try:
                        threading.Thread(target=self.match_face,
                                         args=(frame.copy(),)).start()
                    except ValueError:
                        pass

                with self.lock:
                    if self.face_match:
                        cv2.putText(frame, 'MATCH!', (20, 450),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
                    else:
                        cv2.putText(frame, 'NO MATCH!', (20, 450),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

                cv2.imshow("Video Capture", frame)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    face_recognition = FaceRecognition()
    face_recognition.run()
