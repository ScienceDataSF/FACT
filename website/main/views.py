from django.http import JsonResponse
from django.shortcuts import render
import os
import cv2 as cv
import numpy as np
from django.core.files.storage import FileSystemStorage

def home(request):
    result = None
    if request.method == 'POST' and request.FILES.get('audio'):
        uploaded_file = request.FILES['audio']
        fs = FileSystemStorage(location='main/static/uploads')
        filename = fs.save(uploaded_file.name, uploaded_file)
        file_path = os.path.join('main/static/uploads', filename)

        if uploaded_file.content_type.startswith('audio'):
            result = predict_voice(file_path)
        elif uploaded_file.content_type.startswith('image'):
            result = predict_image(file_path)
        elif uploaded_file.content_type == 'text/plain':
            text_content = uploaded_file.read().decode('utf-8')
            result = predict_text(text_content)
        return JsonResponse({'result': result})
    return render(request, 'home.html',{'result': result})

def predict_voice(audio_path):
    import librosa
    import numpy as np
    import tensorflow as tf

    voice_model = tf.keras.models.load_model('main/models/voicemodel/deepfake_model.h5', compile=False)
    voice_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    y, sr = librosa.load(audio_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc = np.expand_dims(mfcc, axis=-1)
    mfcc = np.resize(mfcc, (1, 128, 109, 1))
    prediction = voice_model.predict(mfcc)

    return 'AI-Generated Voice' if prediction.any() > 0.5 else 'Human Voice'

def predict_text(text_file):
    import joblib
    text_vectorizer = joblib.load('main/models/contentmodel/Model/tfidf_vectorizer.pkl')
    text_model = joblib.load('main/models/contentmodel/Model/llm_text_detection_model.pkl')
    sample_tfidf = text_vectorizer.transform([text_file[:500]])  # Process the first 500 characters for prediction
    prediction = text_model.predict(sample_tfidf)
    if prediction[0] == 'low':
        return "This is not an AI generated text"
    elif prediction[0] == 'med':
        return "The possibility of it being an AI generated text is moderate"
    elif prediction[0] == 'high':
        return "This is an AI generated text"

def predict_image(image_file):
    import numpy as np
    import random
    import cv2 as cv
    from tensorflow.keras.models import load_model
    model_new = load_model(r'C:\Users\Tanmay Somani\OneDrive\Desktop\Programming\!Projects\FACT\website\main\models\imagemodel\AIGeneratedModel.h5')
    categories = ['Real', 'AIGenerated']
    img_size = 48
    img_array = cv.imread(image_file)
    if img_array is None:
        raise FileNotFoundError(f"Image file '{image_file}' not found or could not be loaded.")
    new_array = cv.resize(img_array, (img_size, img_size))
    new_array = new_array / 255.0
    testing_data = []
    classes = categories.index('AIGenerated')  
    testing_data.append([new_array, classes])
    random.shuffle(testing_data)
    X_test = []
    y_test = []
    for features, label in testing_data:
        X_test.append(features)
        y_test.append(label)
    X_test = np.array(X_test).reshape(-1, img_size, img_size, 3)
    y_test = np.array(y_test)
    y_pred = model_new.predict(X_test)
    for arr in y_pred:
        if arr[0] <= 0.5:
            return "This image feels AI Generated"
        else:
            return "This image feels Real"
            
def docs(request):
    return render(request, 'docs.html')

def contact(request):
    return render(request, 'contact.html')
