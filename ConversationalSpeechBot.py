
############## Speech to Text ##############
import numpy as np
import wave
import pyaudio
import time

import noisereduce as nr

from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from transformers import pipeline
import torch

import librosa

LANG_ID = "en"
MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-english"
SAMPLES = 10

speech_to_text_processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
speech_to_text_model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)

def save_audio_file(seconds=5, CHUNK = 1024, FORMAT = pyaudio.paInt16, CHANNELS = 1, RATE = 22050):
    # Record the audio
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    print("Started Recording for" + str(seconds) + "seconds...")
    frames = []
    
    for i in range(0, int(RATE/CHUNK * seconds)):
        data = stream.read(CHUNK)
        frames.append(data)
        
    print("Recording Stopped..")
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    #Saving the audio
    file_name = 'output_' + str(int(time.time())) + '.wav' 
    wf = wave.open("./Dataset/Temp/"+file_name, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    
    return file_name

def speech_to_text(model, processor, audio_file):
    data, sample_rate = librosa.load(audio_file, sr=16000)
    reduced_noise = nr.reduce_noise(y=data, sr=sample_rate)
    input_values = processor(reduced_noise, sampling_rate=sample_rate, return_tensors="pt", padding="longest").input_values
    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    return transcription

# speech_to_text(model, processor, '../Dataset/Temp/output_1663921799.wav')


############## Question Answering    ##############

from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import itertools

question_answer = pipeline("question-answering", model="deepset/electra-base-squad2", top_k=5)

f = open("./Dataset/context.txt", "r")
context = f.read()
f.close()

def max_sum_sim(doc_embedding, candidate_embeddings, candidates, top_n, nr_candidates):
    # Calculate distances and extract keywords
    distances = cosine_similarity(doc_embedding, candidate_embeddings)
    distances_candidates = cosine_similarity(candidate_embeddings, 
                                            candidate_embeddings)

    # Get top_n words as candidates based on cosine similarity
    words_idx = list(distances.argsort()[0][-nr_candidates:])
    words_vals = [candidates[index] for index in words_idx]
    distances_candidates = distances_candidates[np.ix_(words_idx, words_idx)]

    # Calculate the combination of words that are the least similar to each other
    min_sim = np.inf
    candidate = None
    for combination in itertools.combinations(range(len(words_idx)), top_n):
        sim = sum([distances_candidates[i][j] for i in combination for j in combination if i != j])
        if sim < min_sim:
            candidate = combination
            min_sim = sim

    if(candidate):  
        return [words_vals[idx] for idx in candidate]
    else:
        return []

def mmr(doc_embedding, word_embeddings, words, top_n, diversity):

    # Extract similarity within words, and between words and the document
    word_doc_similarity = cosine_similarity(word_embeddings, doc_embedding)
    word_similarity = cosine_similarity(word_embeddings)

    # Initialize candidates and already choose best keyword/keyphras
    keywords_idx = [np.argmax(word_doc_similarity)]
    candidates_idx = [i for i in range(len(words)) if i != keywords_idx[0]]

    for _ in range(top_n - 1):
        # Extract similarities within candidates and
        # between candidates and selected keywords/phrases
        candidate_similarities = word_doc_similarity[candidates_idx, :]
        target_similarities = np.max(word_similarity[candidates_idx][:, keywords_idx], axis=1)

        # Calculate MMR
        mmr = (1-diversity) * candidate_similarities - diversity * target_similarities.reshape(-1, 1)
        if(mmr.size > 0):
            mmr_idx = candidates_idx[np.argmax(mmr)]

            # Update keywords & candidates
            keywords_idx.append(mmr_idx)
            candidates_idx.remove(mmr_idx)

    return [words[idx] for idx in keywords_idx]

def get_keywords_keyBert(sentences, model_name='distilbert-base-nli-mean-tokens', n_gram_range=(1, 2), stop_words="english", top_n=10, diversification=None, nr_candidates=15, diversity=0.5):
    #Get candidate phrases
    count = CountVectorizer(ngram_range=n_gram_range, stop_words=stop_words).fit([sentences])
    candidates = count.get_feature_names_out()
    
    #Load Model
    model = SentenceTransformer(model_name)
    doc_embedding = model.encode([sentences])
    candidate_embeddings = model.encode(candidates)
    
    #Calculate distance between embedding to find similarty
    if(diversification == None):
        distances = cosine_similarity(doc_embedding, candidate_embeddings)
        keywords = [candidates[index] for index in distances.argsort()[0][-top_n:]]
    elif(diversification == 'max_sum_sim'):
        keywords = max_sum_sim(doc_embedding, candidate_embeddings, candidates, top_n=top_n, nr_candidates=nr_candidates)
    elif(diversification == 'mmr'):
        keywords = mmr(doc_embedding, candidate_embeddings, candidates, top_n=top_n, diversity=diversity)
    
    return list(set(keywords))

def get_short_context(question, context):
    keywords = get_keywords_keyBert(question, model_name='all-MiniLM-L6-v2', n_gram_range=(1, 1), diversification='mmr', top_n=3, diversity=0.8)
    possible_context = set()
    for keyword in keywords:
        for sent in context.split('. '):
            if keyword in sent.lower():
                possible_context.add(sent)
    possible_context = list(possible_context)
    possible_context = '. '.join(possible_context)
    
    return possible_context

def get_answers(question, context):
    short_context = get_short_context(question, context)
    qa_input = {
        'question': question,
        'context': context
    }
    res = question_answer(qa_input)
    final_answers = set()
    for r in res:
        if(r['score'] > 0.99):
            final_answers.add(r['answer'])
        
    answers = list(final_answers)
    if(len(answers) == 0):
        return 'No buses available for this route and time'
    else:
        return ', '.join(answers)

########################### Text to Speech ###########################
import torchaudio
from speechbrain.pretrained import Tacotron2
from speechbrain.pretrained import HIFIGAN
from playsound import playsound

tacotron2 = Tacotron2.from_hparams(source="speechbrain/tts-tacotron2-ljspeech", savedir="./Notebooks/tmpdir_tts")
hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-ljspeech", savedir="./Notebooks/tmpdir_vocoder")

def text_to_speech(text):
    mel_output, mel_length, alignment = tacotron2.encode_text(text)
    waveforms = hifi_gan.decode_batch(mel_output)
    file_name = 'result_' + str(int(time.time())) + '.wav' 
    torchaudio.save('./Dataset/Temp/'+file_name, waveforms.squeeze(1), 22050)
    
    return file_name

########################### Runner Function ###########################

import speech_recognition as sr

r = sr.Recognizer();

def record_audio():
    print("Ask the Question...")
    with sr.Microphone() as source:
        audio = r.listen(source)
        voice_data = ''
        try:
            voice_data = r.recognize_google(audio)
        except sr.UnknownValueError:
            return "Sorry I didnt get that"
        except sr.RequestError:
            return "Sorry, My Speech service is down"

        return voice_data;  

if __name__ == '__main__':
    print("Bus Buddy Started")
    # input_file_name = save_audio_file()
    # input_question = speech_to_text(speech_to_text_model, speech_to_text_processor, './Dataset/Temp/'+input_file_name)
    # print(input_question)
    input_question = record_audio()
    print(input_question)
    output_answers = get_answers(input_question, context)
    print(output_answers)
    output_file_name = text_to_speech(output_answers)
    print(output_answers)
    playsound('./Dataset/Temp/'+output_file_name)