import face_recognition
import cv2
import numpy as np

video_capture = cv2.VideoCapture('/Users/mahilans/Downloads/Video_for_traning.mp4')

# sample picture
thor_image = face_recognition.load_image_file("/Users/mahilans/Desktop/OpenCV/mahi/Training_images/Thor_Chris.jpeg")
thor_face_encoding = face_recognition.face_encodings(thor_image)[0]
banner_image = face_recognition.load_image_file("/Users/mahilans/Desktop/OpenCV/mahi/Training_images/banner.jpeg")
banner_face_encoding = face_recognition.face_encodings(banner_image)[0]
hawk_image = face_recognition.load_image_file("/Users/mahilans/Desktop/OpenCV/mahi/Training_images/hawkeye.jpeg")
hawk_face_encoding = face_recognition.face_encodings(hawk_image)[0]
scarlet_image = face_recognition.load_image_file("/Users/mahilans/Desktop/OpenCV/mahi/Training_images/scarlet.jpeg")
scarlet_face_encoding = face_recognition.face_encodings(scarlet_image)[0]

known_face_encodings = [
    thor_face_encoding,
    banner_face_encoding,
    hawk_face_encoding,
    scarlet_face_encoding
]
known_face_names = [
    "Thor",
    "Banner",
    "Hawk",
    "Scarlet"
]

while True:
   
    ret, frame = video_capture.read()

    rgb_frame = frame[:, :, ::-1]

    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    cv2.imshow('Video', frame)

    #Press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
