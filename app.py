from streamlit_webrtc import webrtc_streamer
import av
import cv2
import tensorflow as tf
import numpy as np
import imutils
import pickle
import os

model_path='liveness.model'
le_path='label_encoder.pickle'
encodings='encoded_faces.pickle'
detector_folder='face_detector'
confidence=0.5
args = {'model':model_path, 'le':le_path, 'detector':detector_folder, 
	'encodings':encodings, 'confidence':confidence}

# load the encoded faces and names
print('[INFO] loading encodings...')
with open(args['encodings'], 'rb') as file:
	encoded_data = pickle.loads(file.read())

# load our serialized face detector from disk
print('[INFO] loading face detector...')
proto_path = os.path.sep.join([args['detector'], 'deploy.prototxt'])
model_path = os.path.sep.join([args['detector'], 'res10_300x300_ssd_iter_140000.caffemodel'])
detector_net = cv2.dnn.readNetFromCaffe(proto_path, model_path)
	
# load the liveness detector model and label encoder from disk
liveness_model = tf.keras.models.load_model(args['model'])
le = pickle.loads(open(args['le'], 'rb').read())


class VideoProcessor:		
	def recv(self, frame):
		frm = frame.to_ndarray(format="bgr24")

		
		frm = imutils.resize(frm, width=800)

		
		(h, w) = frm.shape[:2]
		blob = cv2.dnn.blobFromImage(cv2.resize(frm, (300,300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
		
		# pass the blob through the network 
		# and obtain the detections and predictions
		detector_net.setInput(blob)
		detections = detector_net.forward()
		
		# iterate over the detections
		for i in range(0, detections.shape[2]):
			confidence = detections[0, 0, i, 2]
			
			# filter out weak detections
			if confidence > args['confidence']:
				# compute the (x,y) coordinates of the bounding box
				# for the face and extract the face ROI
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype('int')
				
				
				startX = max(0, startX-20)
				startY = max(0, startY-20)
				endX = min(w, endX+20)
				endY = min(h, endY+20)
				
				# extract the face ROI and then preprocess it
				# in the same manner as our training data

				face = frm[startY:endY, startX:endX] # for liveness detection
				# expand the bounding box so that the model can recog easier

				# some error occur here if my face is out of frame and comeback in the frame
				try:
					face = cv2.resize(face, (32,32)) # our liveness model expect 32x32 input
				except:
					break

				name = 'Unknown'
				face = face.astype('float') / 255.0 
				face = tf.keras.preprocessing.image.img_to_array(face)

				
				face = np.expand_dims(face, axis=0)
			
				
				preds = liveness_model.predict(face)[0]
				j = np.argmax(preds)
				label_name = le.classes_[j]
				
				# draw the label and bounding box on the frame
				label = f'{label_name}: {preds[j]:.4f}'
				print(f'[INFO] {name}, {label_name}')
				
				if label_name == 'fake':
					cv2.putText(frm, "Fake Alert!", (startX, endY + 25), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0,0,255), 2)
				
				cv2.putText(frm, name, (startX, startY - 35), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0,130,255),2 )
				cv2.putText(frm, label, (startX, startY - 10),cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)
				cv2.rectangle(frm, (startX, startY), (endX, endY), (0, 0, 255), 4)

		return av.VideoFrame.from_ndarray(frm, format='bgr24')

webrtc_streamer(key="key", video_processor_factory=VideoProcessor,rtc_configuration={ "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},sendback_audio=False, video_receiver_size=1)