# FACT: AI Content Predictor

## Overview
FACT (Fake AI Content Tracker) is a web application built using Django that helps users detect AI-generated content in text, images, and audio. It utilizes machine learning models to classify input content as AI-generated or real.

## Link to the Project
This might load a little slow due to render but will shift to another platform.
([Project](https://fact-wku3.onrender.com/))

## Features
- **AI Text Detection**: Uses a trained TF-IDF and classification model to determine whether a given text is AI-generated.
- **AI Image Detection**: Employs a deep learning model to classify images as real or AI-generated.
- **AI Voice Detection**: Analyzes audio files to distinguish between human and AI-generated voices.
- **Web Interface**: Provides an intuitive UI for uploading and analyzing content.

## Technologies Used
- **Backend**: Django, Django REST Framework
- **Frontend**: HTML, CSS, JavaScript
- **Machine Learning**: TensorFlow/Keras, Librosa, Joblib, OpenCV
- **Database**: SQLite
- **Deployment**: Gunicorn, Render

## Installation
### Prerequisites
Ensure you have Python (>=3.7) and pip installed.

### Steps
1. Clone the repository:
   ```sh
   git clone https://github.com/your-repo/fact-ai-content-detector.git
   cd fact-ai-content-detector
   ```
2. Create a virtual environment and activate it:
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\\Scripts\\activate`
   ```
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
4. Apply migrations and start the server:
   ```sh
   python manage.py migrate
   python manage.py runserver
   ```
5. Open `http://127.0.0.1:8000/` in your browser.

## Usage
1. Visit the home page.
2. Upload a text, image, or audio file.
3. Click submit and view the results.

## Deployment
This project is configured for deployment using **Render**.
- The `render.yaml` file defines the deployment setup.
- The `Procfile` is used to run Gunicorn.

## Project Structure
```
sciencedatasf-fact/
├── Statistics/               # Stores function logs
│   └── function_log.csv
├── website/                  # Main Django app
│   ├── db.sqlite3            # SQLite database
│   ├── manage.py             # Django management script
│   ├── main/                 # App directory
│   │   ├── models/           # ML models directory
│   │   │   ├── contentmodel/
│   │   │   ├── imagemodel/
│   │   │   ├── voicemodel/
│   │   ├── static/           # Static files
│   │   ├── templates/        # HTML templates
│   │   ├── views.py          # Application logic
│   ├── website/              # Django project settings
│   │   ├── settings.py
│   │   ├── urls.py
│   │   ├── wsgi.py
├── requirements.txt          # Dependencies
├── render.yaml               # Deployment config
└── Procfile                  # Server run command
```

## Contributors
- **Tanmay Somani** ([LinkedIn](https://www.linkedin.com/in/tcode/))
- **Varun Pandya**  ([LinkedIn](https://www.linkedin.com/in/varun-pandya/)
- **Rishabh Jaiswal**

## License
This project is open-source and available under the MIT License.



