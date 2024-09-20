## Multimodal LLM for Video Advertisement Understanding and Classification

## Table of Contents
- Introduction
- Dataset
- Data Preprocessing
- Approach
    1. Naive BERT Approach
    2. Retrieval Augmented Generation (RAG) Approach
- Results
- Evaluation Metrics
- Technologies Used
- How to Run
- Future Work
- Contributing
- License

## Introduction
This project leverages artificial intelligence to analyze video advertisements and answer 21 binary (yes/no) questions based on text descriptions and speech captions. The key goal is to document the results, compute precision, recall, F1-score, and analyze overall performance using transformer-based models and retrieval-augmented generation (RAG). A web application was also developed for querying and retrieving relevant information from the ads.

## Dataset
- Videos: 150 video advertisements of varying lengths from different companies.
- Textual Data: Includes ad campaign descriptions, transcriptions, and on-screen text extracted from videos.
- Ground Truth: Answers provided by multiple coders for 21 binary yes/no questions using a majority vote approach to determine ground truth.

## Data Preprocessing
Merged creative data with ground truth.
Generated a long-form dataset for each (speech, question) pair.
Processed the data into structured columns including ID, speech, description, question, and label.

## Approach

 1. **Naive BERT Approach**
I used pre-trained BERT models to process video transcriptions and answer the binary questions. The pipeline:

- Extracts audio from video (using MoviePy and Whisper).
- Transcribes audio to text.
- Uses BERT for tokenization and embedding.
- Computes cosine similarity to generate yes/no predictions for each question.

2. **RAG Approach**
This approach incorporates Retrieval Augmented Generation (RAG) by:

- Transcribing video content and storing text vectors using FAISS for efficient semantic search.
- Using vector-based similarity searches to answer binary questions based on video descriptions.
- Adding a caching mechanism to improve response times for repeated queries.

## Results

1. Naive BERT
- Precision: High for questions related to specific products/services and brand visibility.
- Mixed Performance: Moderate for questions involving humor and relatable characters.
- Low Performance: Struggled with subjective questions like sensory stimulation and visually pleasing.

2. RAG
- High Precision & Recall: Questions related to specific product/service mentions and brand visibility.
- Low Precision: False positives for questions like "call to go online" and "sense of urgency".

3. Evaluation Metrics
- Precision: 0.50
- Recall: 0.72
- F1 Score: 0.58
- Accuracy: 0.51
- ROC AUC: 0.53
- Agreement Percentage: 52%

## Technologies Used

- BERT: For text embedding and classification.
- Faiss: For vector storage and similarity search.
- OpenAI Whisper: For speech-to-text conversion.
- MoviePy: For video-to-audio extraction.
- Streamlit: For the web-based user interface.
- Transformers Library: For model implementation.