import streamlit as st
import assemblyai as aai
from transformers import pipeline
import tempfile
import openai
import os

# Configuration de la page Streamlit
st.set_page_config(layout="wide")

# Configuration des clés API
os.environ["TOKENIZERS_PARALLELISM"] = "false"
aai.settings.api_key = st.sidebar.text_input('Veuillez insérer la clée fournie pour transcription', type='password')
openai.api_key = st.sidebar.text_input('Veuillez insérer la clée fournie pour démonstration', type='password')

st.sidebar.image("logo2.jpg", use_column_width=True)

def transcribe_audio(audio_path):
    transcriber = aai.Transcriber()
    config = aai.TranscriptionConfig(language_code="fr", speaker_labels=True, speakers_expected=2)
    transcript = transcriber.transcribe(audio_path, config)
    return transcript

sentiment_analysis = pipeline(
  "sentiment-analysis",
  framework="pt",
  model="lxyuan/distilbert-base-multilingual-cased-sentiments-student" #SamLowe/roberta-base-go_emotions  #lxyuan/distilbert-base-multilingual-cased-sentiments-student
)

def analyze_sentiment_voice(text):
    results = sentiment_analysis(text)
    sentiment_label = results[0]['label']
    return sentiment_label

def analyze_emotion(text):
   try:
       content = f"Please analyze the following text to detect the underlying emotion. Return the detected emotion in French (e.g., 'Neutre', 'Frustration', 'Colère', etc.). If no specific emotion is detected, please respond with 'Neutre'. The text for analysis is: \n{text}"
       messages = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": content}]
       response = openai.ChatCompletion.create(
           model="gpt-3.5-turbo-16k",
           messages=messages,
           max_tokens=100
       )
       emotion_result = response['choices'][0]['message']['content']
       emotion = emotion_result.split(":")[-1].strip()
       return emotion
   except Exception as e:
       st.error(f"Erreur lors de l'analyse de l'émotion : {e}")
       return None

def display_transcription(transcript):
    for utterance in transcript.utterances:
        st.write(f"<span style='color: #922B21;'>Speaker {utterance.speaker}:</span> {utterance.text}", unsafe_allow_html=True)

def main():
    st.markdown("<h1 style='text-align:center;font-size:xx-large;color: #B01817;'>Transcription audio et analyse émotionnelle</h1>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Téléverser un fichier audio", type=["mp3", "wav"])
    button_col1, button_col2, button_col3 = st.columns(3)

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            audio_path = tmp_file.name
        st.audio(open(audio_path, "rb").read(), format="audio/wav")

        if button_col1.button("Transcription") :
            transcript = transcribe_audio(audio_path)
            display_transcription(transcript)

        if button_col2.button("Émotion basée sur le texte") and uploaded_file:
            transcript = transcribe_audio(audio_path)
            for utterance in transcript.utterances:
                st.write(f"<span style='color: blue;'>Speaker {utterance.speaker}:</span> {utterance.text}", unsafe_allow_html=True)
                sentiment = analyze_emotion(utterance.text)
                st.write("Emotion détectée : ", sentiment)

        if button_col3.button("Émotion basée sur la voix") and uploaded_file:
                transcript = transcribe_audio(audio_path)
                for utterance in transcript.utterances:
                    st.write(f"<span style='color: blue;'>Speaker {utterance.speaker}:</span> {utterance.text}", unsafe_allow_html=True)
                    sentiment0 = analyze_sentiment_voice(utterance.text)
                    st.write("Sentiment : ", sentiment0)
    else:
        st.write("Veuillez uploader un fichier audio pour commencer la transcription.")    

if __name__ == "__main__":
    main()
