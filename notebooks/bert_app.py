import streamlit as st
from transformers import BertTokenizer, BertModel, pipeline
import torch
import numpy as np
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import moviepy.editor as mp
import speech_recognition as sr
import os

os.environ['PATH'] += os.pathsep + 'C:/Program Files/ffmpeg/bin'


# Initialize BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')


def transcribe_audio(audio_file_path):
    """
    Transcribes audio from a specified file path using a pre-trained ASR model.

    Args:
    audio_file_path (str): Path to the audio file for transcription.

    Returns:
    str: Transcription text of the audio file.

    This function uses the Hugging Face `pipeline` to perform automatic speech recognition (ASR)
    on the audio file located at `audio_file_path`. It returns the transcription text of the audio.
    """
    transcriber = pipeline(task="automatic-speech-recognition", model="openai/whisper-small")
    transcription_results = transcriber(audio_file_path)
    transcription_text = transcription_results.get('text', "No transcription results found.")
    return transcription_text  

# Function to process video and transcribe audio
def transcribe_audio_from_video(video_file_path):
    """
    Transcribes audio from a video file using automatic speech recognition (ASR).

    Args:
    video_file_path (str): Path to the video file for transcription.

    Returns:
    str: Transcription text of the video's audio.

    This function extracts audio from the video file located at `video_file_path`, saves it as
    a .wav file, performs transcription using the `transcribe_audio` function, and returns the
    transcription text.
    """
    
    video = mp.VideoFileClip(video_file_path)
    audio_file = video.audio
    audio_path = os.path.join("audio_files", os.path.basename(video_file_path).replace('.mp4', '.wav'))
    audio_file.write_audiofile(audio_path)
    
    # Transcribe the audio using the Whisper model
    transcription = transcribe_audio(audio_path)
    
    # Print transcription
    print("Transcription Results:", transcription)
    
    return transcription


# Predefined set of questions
questions = [
    "Is there a call to go online (e.g., shop online, visit the Web)?",
    "Is there online contact information provided (e.g., URL, website)?",
    "Is there a visual or verbal call to purchase (e.g., buy now, order now)?",
    "Does the ad portray a sense of urgency to act (e.g., buy before sales ends, order before ends)?",
    "Is there an incentive to buy (e.g., a discount, a coupon, a sale or 'limited time offer')?",
    "Is there offline contact information provided (e.g., phone, mail, store location)?",
    "Is there mention of something free?",
    "Does the ad mention at least one specific product or service (e.g., model, type, item)?",
    "Is there any verbal or visual mention of the price?",
    "Does the ad show the brand (logo, brand name) or trademark (something that most people know is the brand) multiple times?",
    "Does the ad show the brand or trademark exactly once at the end of the ad?",
    "Is the ad intended to affect the viewer emotionally, either with positive emotion or negative emotion?",
    "Does the ad give you a positive feeling about the brand?",
    "Does the ad have a story arc, with a beginning and an end?",
    "Does the ad have a reversal of fortune, where something changes for the better or worse?",
    "Does the ad have relatable characters?",
    "Is the ad creative/clever?",
    "Is the ad intended to be funny?",
    "Does this ad provide sensory stimulation?",
    "Is the ad visually pleasing?",
    "Does the ad have cute elements like animals, babies, animated characters, etc?"
]

question_embeddings = []
for question in questions:
    encoded_input = tokenizer(question, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        output = model(**encoded_input)
        embeddings = output.last_hidden_state.mean(dim=1).squeeze().numpy()
    question_embeddings.append(embeddings)

# Convert to numpy array for consistency
question_embeddings = np.array(question_embeddings)

def process_text_and_predict(user_input_text):
    """
    Processes user input text and predicts answers using a pre-trained model and cosine similarity.

    Args:
    user_input_text (str): User input text to process and predict answers.

    Returns:
    dict: A dictionary mapping questions to predicted answers ("YES" or "NO").

    This function tokenizes and encodes the `user_input_text`, computes embeddings using a
    pre-trained model, performs vector search using cosine similarity against `question_embeddings`,
    determines answers based on similarity scores using a defined threshold, and returns a dictionary
    mapping each question to its predicted answer ("YES" or "NO").
    """
    # Tokenize and encode the user input text
    encoded_user_input = tokenizer(user_input_text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        output = model(**encoded_user_input)
        user_input_embedding = output.last_hidden_state.mean(dim=1).squeeze().numpy()

    # Perform vector search using cosine similarity
    similarity_scores = cosine_similarity([user_input_embedding], question_embeddings).flatten()

    # Define a threshold for decision 
    threshold = 0.6

    # Determine answers based on similarity scores
    predicted_answers = []
    for score, question in zip(similarity_scores, questions):
        if score >= threshold:
            predicted_answers.append("YES")  # YES
        else:
            predicted_answers.append("NO")  # NO

    return dict(zip(questions, predicted_answers))

# Title of the app
st.title('Speech/Video Processing Application')

col1, col2 = st.columns([1, 2])


# User selects the input type
input_type = st.selectbox("Choose input type:", ["Text/Speech", "Video"])

if input_type == "Text/Speech":
    user_input_text = st.text_area("Enter the text or speech:")
    if st.button("Process Text/Speech"):
        st.write("Processing text/speech...")
        answers = process_text_and_predict(user_input_text)
        
        # Display questions and predicted answers  
        for question in questions:
                st.write(f"Question: {question}")
                st.write(f"Predicted Answer: {answers[question]}")
                st.write("\n") 
        st.title('Thank you!')


elif input_type == "Video":
    uploaded_video = st.file_uploader("Upload a video file:", type=["mp4", "avi", "mov"])
    if uploaded_video is not None:
        # Create directories if they do not exist
        if not os.path.exists("uploaded_videos"):
            os.makedirs("uploaded_videos")
        if not os.path.exists("audio_files"):
            os.makedirs("audio_files")
        
        save_path = os.path.join("uploaded_videos", uploaded_video.name)
        
        # Save the uploaded video file
        with open(save_path, "wb") as f:
            f.write(uploaded_video.read())
        
        if st.button("Process Video"):
            st.write("Processing video...")
            progress_bar = st.progress(0)
            progress_bar.progress(25)
            transcription_text = transcribe_audio_from_video(save_path)
            progress_bar.progress(50)
            # st.write("Transcription:", transcription_text)
            answers = process_text_and_predict(transcription_text)
            progress_bar.progress(100)
           
            
            st.video(save_path)
            st.markdown("**Transcription:**")
            st.write(transcription_text)
            st.markdown("**Questions and Predicted Answers:**")
            for question in questions:
                st.write(f"Question: {question}")
                st.write(f"Answer: {answers[question]}")
                st.write("\n")

            st.title('Thank you!')

