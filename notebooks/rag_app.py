import streamlit as st
from transformers import BertTokenizer, BertModel, pipeline
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import moviepy.editor as mp
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, AutoModel
import torch
import faiss
import ast
from transformers import BertTokenizer, BertModel
import os


os.environ['PATH'] += os.pathsep + 'C:/Program Files/ffmpeg/bin'

# Initialize BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')


# Function to split text into chunks
def split_text_into_chunks(text, chunk_size=80):
    """
    Splits the input text into smaller chunks of specified size.

    Args:
    text (str): The text to be split into chunks.
    chunk_size (int, optional): The size of each chunk. Defaults to 80.

    Returns:
    list: A list containing the text chunks.
    
    This function takes a string `text` and splits it into chunks of size `chunk_size`.
    It iterates through the text, appending each chunk of the specified size to the 
    `chunks` list. The result is a list of text chunks, where each chunk is at most 
    `chunk_size` characters long.
    """
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i+chunk_size])
    return chunks

# Function to generate embeddings for text chunks
def generate_embeddings(chunks):
    """
    Generates embeddings for each chunk of text using a Sentence-BERT model.

    Args:
    chunks (list): List of strings where each string represents a chunk of text.

    Returns:
    numpy.ndarray: 2D array of embeddings where each row corresponds to the embedding of a chunk.

    This function initializes a Sentence-BERT model ('paraphrase-MiniLM-L6-v2')
    and computes embeddings for each chunk of text in the input list `chunks`. The embeddings
    are computed using the Sentence-BERT model and stored in a list. These embeddings are then
    stacked vertically to form a 2D numpy array. Each row in the array represents the embedding
    vector of a chunk of text.
    """
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    embeddings = []
    for chunk in chunks:
        chunk_embedding = model.encode(chunk)
        embeddings.append(chunk_embedding)
    embeddings = np.vstack(embeddings)
    return embeddings

# Function to build FAISS index
def build_faiss_index(embeddings, dimension=384):
    """
    Builds a FAISS index for fast similarity search using the given embeddings.

    Args:
    embeddings (numpy.ndarray): 2D array of embeddings where each row represents the embedding vector of a chunk of text.
    dimension (int, optional): Dimensionality of the embedding vectors. Default is 384.

    Returns:
    faiss.IndexFlatL2: FAISS index object configured with L2 distance metric, populated with the provided embeddings.

    This function initializes a FAISS index with the specified dimensionality and adds the given embeddings to it.
    The FAISS index is optimized for efficient similarity search based on L2 distance metric. It returns the initialized
    FAISS index object ready for use in similarity searches.
    """
    index = faiss.IndexFlatL2(dimension)  # L2 distance metric
    index.add(embeddings)
    return index

# Function to retrieve relevant documents
def retrieve_documents(user_question, index, chunks):
    """
    Retrieves relevant documents (chunks of text) based on the user question using a pre-built FAISS index.

    Args:
    user_question (str): User query for retrieving relevant documents.
    index (faiss.IndexFlatL2): Pre-built FAISS index object for similarity search.
    chunks (list): List of text chunks corresponding to the embeddings used for indexing.

    Returns:
    list: List of relevant text chunks retrieved based on the similarity search using FAISS.

    This function encodes the user question using a Sentence-BERT model, then performs a vector search using
    FAISS to retrieve the closest embeddings to the query. It returns the corresponding text chunks that are
    most relevant to the user question based on the similarity scores obtained from FAISS.
    """
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    query_embedding = model.encode(user_question).astype(np.float32).reshape(1, -1)
    D, I = index.search(query_embedding, 5)  # Retrieve top 5 closest embeddings
    relevant_docs = [chunks[i] for i in I.flatten()]
    return relevant_docs

# Function to perform semantic search
def semantic_search(user_question, relevant_docs):
    """
    Performs semantic search to determine if there are relevant documents (chunks of text) that match the user question.

    Args:
    user_question (str): User query to find relevant documents.
    relevant_docs (list): List of relevant text chunks retrieved from the FAISS index.

    Returns:
    str: "Yes" if there are relevant documents with cosine similarity above a threshold, otherwise "No".

    This function encodes the user question and relevant documents using a Sentence-BERT model,
    calculates cosine similarity scores, and determines if any relevant documents match the user question
    based on a predefined similarity threshold. It returns "Yes" if a relevant match is found, otherwise "No".
    """
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    query_embedding = model.encode(user_question, convert_to_tensor=True)
    doc_embeddings = model.encode(relevant_docs, convert_to_tensor=True)
    cos_sim_scores = util.pytorch_cos_sim(query_embedding, doc_embeddings)
    if cos_sim_scores.max() > 0.25:  # Adjust threshold as needed
        return "Yes"
    else:
        return "No"

# Function to process a corpus
def process_corpus(corpus, questions):
    """
    Processes a corpus of text, performs semantic search for each question, and returns answers.

    Args:
    corpus (str): Full text corpus to process.
    question_embeddings (list): List of embeddings for each question.
    questions (list): List of questions to answer based on the corpus.

    Returns:
    dict: A dictionary where keys are questions and values are answers ("Yes" or "No").

    This function splits the corpus into text chunks, generates embeddings for each chunk,
    builds a FAISS index for efficient retrieval, and performs semantic search to answer
    each question based on the relevant chunks of text. It returns a dictionary of questions
    mapped to their corresponding answers.
    """
    chunks = split_text_into_chunks(corpus)
    embeddings = generate_embeddings(chunks)
    index = build_faiss_index(embeddings, dimension=embeddings.shape[1])
    results = {}
    for question in questions:
        relevant_docs = retrieve_documents(question, index, chunks)
        answer = semantic_search(question, relevant_docs)
        results[question] = answer
    return results

# Function to transcribe audio
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
    audio_path = os.path.join("audio_files", os.path.basename(video_file_path).replace('.mp4', '.wav'))
    video.audio.write_audiofile(audio_path)
    transcription = transcribe_audio(audio_path)
    return transcription

# Define the questions
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


# Streamlit UI
st.title('RAG Chatbot for Video Analysis')
st.header('Upload a video file and ask questions about its content')

# Ensure Streamlit session state is used to store intermediate results
if 'transcription_text' not in st.session_state:
    st.session_state['transcription_text'] = None

if 'answers' not in st.session_state:
    st.session_state['answers'] = None  


# User selects the input type
input_type = st.selectbox("Choose input type:", ["Text/Speech", "Video"])

if input_type == "Text/Speech":
    user_input_text = st.text_area("Enter the text or speech:")
    if st.button("Process Text/Speech"):
        st.write("Processing text/speech...")
        answers = process_corpus(user_input_text, questions)
        st.session_state['answers'] = process_corpus(user_input_text, questions) 
        st.write("Predicted Answers:", answers)
            


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
            # answers = process_text_and_predict(transcription_text)
            progress_bar.progress(100)
            st.video(save_path)
            st.markdown("**Transcription:**")
            # st.write(transcription_text)
            st.session_state['transcription_text'] = transcription_text
            st.write("Transcription Results:", transcription_text)
            # Process corpus to get answers
            st.session_state['answers'] = process_corpus(transcription_text, questions) 

# Select question mode
question_mode = st.radio("Choose how to ask your question:", ("Select from dropdown", "Type your question"))

if question_mode == "Select from dropdown":
    selected_question = st.selectbox("Select a question:", questions)
else:
    selected_question = st.text_input("Type your question:")

if st.button("Get Answer"):
        if st.session_state['answers'] is not None:
            predicted_answer = st.session_state['answers'].get(selected_question, "No")
            st.write(f"Question: {selected_question}")
            st.write(f"Predicted Answer: {predicted_answer}")
        else:
            st.write("Please upload and process a video first.")

st.title('Thank you!')

