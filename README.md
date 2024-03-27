# EmotionScore-Emotionserkennung-f-r-personalisierte-Filmempfehlungen
EmotionScore analysiert die Gesichtsausdrücke von Zuschauern während des Filmkonsums und gibt personalisierte Empfehlungen basierend auf ihren emotionalen Reaktionen.
import cv2
import numpy as np
from keras.models import load_model

# Laden des vortrainierten Modells zur Emotionserkennung
emotion_model = load_model('pfad_zum_emotionserkennungsmodell.h5')

# Zuordnung der Emotionen zu Filmgenres
emotion_genre_mapping = {
    'happy': 'Komödie',
    'sad': 'Drama',
    'angry': 'Action',
    'surprise': 'Thriller',
    'neutral': 'Dokumentarfilm',
}

# Funktion zur Vorhersage der Emotion aus dem Gesichtsbild
def predict_emotion(face_image_gray):
    resized_img = cv2.resize(face_image_gray, (48, 48), interpolation=cv2.INTER_AREA)
    image_pixels = np.expand_dims(resized_img, axis=0)
    image_pixels = np.expand_dims(image_pixels, axis=-1)
    
    predictions = emotion_model.predict(image_pixels)
    max_index = np.argmax(predictions[0])
    return list(emotion_genre_mapping.keys())[max_index]

# Hauptfunktion zur Verarbeitung der Videodaten
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    emotion_counts = {emotion: 0 for emotion in emotion_genre_mapping.keys()}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            face_image_gray = gray[y:y+h, x:x+w]
            predicted_emotion = predict_emotion(face_image_gray)
            emotion_counts[predicted_emotion] += 1

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
    # Analyse der gesammelten Emotionsdaten und Erstellung von Empfehlungen
    favorite_emotion = max(emotion_counts, key=emotion_counts.get)
    recommended_genre = emotion_genre_mapping[favorite_emotion]
    print(f"Basierend auf Ihren emotionalen Reaktionen empfehlen wir Filme aus dem Genre: {recommended_genre}")

# Videoverarbeitung starten (Pfad zum Videodatei ersetzen)
process_video('pfad_zum_video.mp4')
