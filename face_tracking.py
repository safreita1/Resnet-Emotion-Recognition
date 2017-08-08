import cv2
import numpy as np
import tflearn
import operator
from tflearn.data_preprocessing import ImagePreprocessing
from collections import deque


class EmotionRecognition:
    def __init__(self):
        # Create emotion queue of last 'x' emotions to smooth the output
        self.emotion_queue = deque(maxlen=10)
        self.n = 5
        self.model = None

    def smooth_emotions(self, prediction):
        emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
        emotion_values = {'Angry': 0.0, 'Disgust': 0.0, 'Fear': 0.0, 'Happy': 0.0, 'Sad': 0.0, 'Surprise': 0.0, 'Neutral': 0.0}

        emotion_probability, emotion_index = max((val, idx) for (idx, val) in enumerate(prediction[0]))
        emotion = emotions[emotion_index]

        # Append the new emotion and if the max length is reached pop the oldest value out
        self.emotion_queue.appendleft((emotion_probability, emotion))

        # Iterate through each emotion in the queue and create an average of the emotions
        for pair in self.emotion_queue:
            emotion_values[pair[1]] += pair[0]

        # Select the current emotion based on the one that has the highest value
        average_emotion = max(emotion_values.iteritems(), key=operator.itemgetter(1))[0]

        return average_emotion

    def process_image(self, roi_gray, img):
        image_scaled = np.array(cv2.resize(roi_gray, (48, 48)), dtype=float)
        image_processed = image_scaled.flatten()
        image_processed = image_processed.reshape([-1, 48, 48, 1])

        prediction = self.model.predict(image_processed)
        emotion = self.smooth_emotions(prediction)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, "Emotion: " + emotion, (50, 450), font, 1, (255, 255, 255), 2, cv2.CV_AA)
        cv2.imshow('img', img)

    def run(self):
        # Real-time pre-processing of the image data
        img_prep = ImagePreprocessing()
        img_prep.add_featurewise_zero_center()
        img_prep.add_featurewise_stdnorm()

        # Real-time data augmentation
        img_aug = tflearn.ImageAugmentation()
        img_aug.add_random_flip_leftright()
        # img_aug.add_random_crop([48, 48], padding=8)

        # Building Residual Network
        net = tflearn.input_data(shape=[None, 48, 48, 1], data_preprocessing=img_prep, data_augmentation=img_aug)
        net = tflearn.conv_2d(net, nb_filter=16, filter_size=3, regularizer='L2', weight_decay=0.0001)
        net = tflearn.residual_block(net, self.n, 16)
        net = tflearn.residual_block(net, 1, 32, downsample=True)
        net = tflearn.residual_block(net, self.n - 1, 32)
        net = tflearn.residual_block(net, 1, 64, downsample=True)
        net = tflearn.residual_block(net, self.n - 1, 64)
        net = tflearn.batch_normalization(net)
        net = tflearn.activation(net, 'relu')
        net = tflearn.global_avg_pool(net)

        # Regression
        net = tflearn.fully_connected(net, 7, activation='softmax')
        mom = tflearn.Momentum(learning_rate=0.1, lr_decay=0.0001, decay_step=32000, staircase=True, momentum=0.9)
        net = tflearn.regression(net, optimizer=mom,
                                 loss='categorical_crossentropy')

        self.model = tflearn.DNN(net, checkpoint_path='models/model_resnet_emotion',
                            max_checkpoints=10, tensorboard_verbose=0,
                            clip_gradients=0.)

        self.model.load('current_model/model_resnet_emotion-42000')

        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        cap = cv2.VideoCapture(0)

        while True:
            ret, img = cap.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                roi_gray = gray[y:y + h, x:x + w]
                roi_color = img[y:y + h, x:x + w]
                self.process_image(roi_gray, img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    emotion_recognition = EmotionRecognition()
    emotion_recognition.run()