import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import math

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
offset = 20
imgSize = 300

labels = ["Call", "ok fine", "There is Gun", "Stop", "Alone", "I want food", "Thief", "not fine"]

gesture_stats = {label: {"true_positives": 0, "false_positives": 0, "false_negatives": 0} for label in labels}
correct_predictions = 0
total_predictions = 0
confidence_threshold = 0.5

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        if imgCrop.shape[0] > 0 and imgCrop.shape[1] > 0:
            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                imgWhite[:, math.ceil((imgSize - wCal) / 2):wCal + math.ceil((imgSize - wCal) / 2)] = imgResize

            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgWhite[math.ceil((imgSize - hCal) / 2):hCal + math.ceil((imgSize - hCal) / 2), :] = imgResize

            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            confidence_value = prediction[index]

            if confidence_value >= confidence_threshold:
                cv2.putText(imgOutput, labels[index], (x, y - 20), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 2)
                cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (255, 0, 255), 4)
                total_predictions += 1

                ground_truth_labels = ["Call", "ok fine", "There is Gun", "Stop", "Alone", "I want food", "Thief", "not fine"] # Set the list of ground truth labels for the current hand gesture
                if labels[index] in ground_truth_labels:
                    correct_predictions += 1
                    gesture_stats[labels[index]]["true_positives"] += 1
                else:
                    for label in ground_truth_labels:
                        gesture_stats[label]["false_negatives"] += 1
                    gesture_stats[labels[index]]["false_positives"] += 1

                # Calculate and print precision, recall, and accuracy for each hand gesture
                for label in labels:
                    true_positives = gesture_stats[label]["true_positives"]
                    false_positives = gesture_stats[label]["false_positives"]
                    false_negatives = gesture_stats[label]["false_negatives"]

                    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
                    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
                    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

                    print(f"{label}: Precision={precision:.2f}, Recall={recall:.2f}, Accuracy={accuracy:.2f}")

    cv2.imshow("Image", imgOutput)
    key = cv2.waitKey(1)






