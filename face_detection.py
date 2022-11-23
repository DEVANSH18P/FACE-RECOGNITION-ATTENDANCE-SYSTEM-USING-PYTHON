#importing librarys
import cv2
import numpy as npy
import face_recognition as face_rec

#function
def resize(img,size) :
    width = int(img.shape[1]*size)
    height = int(img.shape[0] * size)
    dimension =(width,height)
    return cv2.resize(img , dimension, interpolation= cv2.INTER_AREA)

# img declaration
devansh=face_rec.load_image_file('sampleing/devansh.jpg')
devansh = cv2.cvtColor(devansh,cv2.COLOR_BGR2RGB)
devansh = resize(devansh, 0.50)
devansh_sample=face_rec.load_image_file('sampleing/elon.jpg')
devansh_sample = cv2.cvtColor(devansh_sample,cv2.COLOR_BGR2RGB)
devansh_sample = resize(devansh_sample, 0.50)

# findng the face location

faceLocation_devansh = face_rec.face_locations(devansh)[0]
encode_devansh = face_rec.face_encodings(devansh)[0]
cv2.rectangle(devansh, (faceLocation_devansh[3], faceLocation_devansh[0]), (faceLocation_devansh[1], faceLocation_devansh[2]), (255,0, 255), 3)

faceLocation_devansh_sample = face_rec.face_locations(devansh_sample)[0]
encode_devansh_sample = face_rec.face_encodings(devansh_sample)[0]

cv2.rectangle(devansh_sample, (faceLocation_devansh_sample[3], faceLocation_devansh_sample[0]),
              (faceLocation_devansh_sample[1], faceLocation_devansh_sample[2]), (255, 0, 255), 3)

results = face_rec.compare_faces([encode_devansh], encode_devansh_sample)
print(results)

cv2.putText(devansh_sample, f'{results}', (50,50), cv2.FONT_HERSHEY_COMPLEX, 1,(0,0,255), 2)
cv2.imshow('main_img', devansh)
cv2.imshow('test_img', devansh_sample)
cv2.waitKey(0)
cv2.destroyAllWindows()