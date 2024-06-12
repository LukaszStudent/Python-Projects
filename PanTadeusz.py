import pyaudio
from collections import Counter
import speech_recognition as sr
import os 
from pydub import AudioSegment
from pydub.silence import split_on_silence
import soundfile as sf
import matplotlib.pyplot as plt

plik_rozpoznany_txt_1=('PanTadeuszFragment_1.wav')
plik_rozpoznany_txt_2=('PanTadeuszFragment_2.wav')
plik_rozpoznany_txt_3=('PanTadeuszFragment_3.wav')

r = sr.Recognizer()

def get_large_audio_transcription(path,miejsce_zapisu):
    """
    Splitting the large audio file into chunks
    and apply speech recognition on each of these chunks
    """
    sound = AudioSegment.from_wav(path)  
    chunks = split_on_silence(sound,
        min_silence_len = 500,
        silence_thresh = sound.dBFS-14,
        keep_silence=500,
    )
    folder_name = "audio-chunks"
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)
    whole_text = ""
    for i, audio_chunk in enumerate(chunks, start=1):
        chunk_filename = os.path.join(folder_name, f"chunk{i}.wav")
        audio_chunk.export(chunk_filename, format="wav")
        with sr.AudioFile(chunk_filename) as source:
            audio_listened = r.record(source)
            try:
                text = r.recognize_google(audio_listened, language = 'pl_PL')
            except sr.UnknownValueError as e:
                print("Error:", str(e))
            else:
                text = f"{text.capitalize()}. "
                print(chunk_filename, ":", text)
                whole_text += text
    f = open(miejsce_zapisu, "w")
    f.write(whole_text)
    f.close()
    #return whole_text



def dokladnosc_rozpoznawania_mowy(plik1,plik2):
    f1 = open(plik1, "r",encoding='utf-8')
    f2 = open(plik2, "r")

    all_f1_tab,all_f2_tab=[],[]
    for line1 in f1:
        line1=line1.replace('\n',' ')
        line1=line1.lower()
        rozdzielone_f1=line1.split(' ')
        while '' in rozdzielone_f1:
            rozdzielone_f1.remove('')
        all_f1_tab+=rozdzielone_f1
    f1.close()

    for line2 in f2:
        line2=line2.replace('\n',' ')
        line2=line2.lower()
        rozdzielone_f2=line2.split(' ')
        while '' in rozdzielone_f2:
            rozdzielone_f2.remove('')
        all_f2_tab+=rozdzielone_f2
    f2.close() 

    print(all_f1_tab)
    print(len(all_f1_tab))
    print()
    print(all_f2_tab)
    print(len(all_f2_tab))

    przetoworzony_tekst_f1,przetoworzony_tekst_f2=[],[]
    for slowo in all_f1_tab:
        nowe_slowo=''
        for char in slowo:
            if(ord(char)<65 or (ord(char)>122 and ord(char)<192) or (ord(char)>90 and ord(char)<97)):
                #print('Bledny znak w slowie: ',slowo,'-> ',char)
                pass
            else:
                nowe_slowo+=char
        #print(nowe_slowo)
        przetoworzony_tekst_f1.append(nowe_slowo)

    for slowo in all_f2_tab:
        nowe_slowo=''
        for char in slowo:
            if(ord(char)<65 or (ord(char)>122 and ord(char)<192) or (ord(char)>90 and ord(char)<97)):
                #print('Bledny znak w slowie: ',slowo,'-> ',char)
                pass
            else:
                nowe_slowo+=char
        #print(nowe_slowo)
        przetoworzony_tekst_f2.append(nowe_slowo)


    unikatowe_f1=Counter(przetoworzony_tekst_f1).keys()

    liczba_slow_w_tekscie=len(przetoworzony_tekst_f1)
    licznik=0
    for i in przetoworzony_tekst_f2:
        for j in unikatowe_f1:
            if(i==j):
                licznik+=1
    print('takie same',licznik)
    rozpoznany_tekst_procenty=(licznik/liczba_slow_w_tekscie)*100
    print()

    print(przetoworzony_tekst_f1)
    print()
    print(przetoworzony_tekst_f2)
    return rozpoznany_tekst_procenty

get_large_audio_transcription(plik_rozpoznany_txt_1,'plik_1.txt')
get_large_audio_transcription(plik_rozpoznany_txt_2,'plik_2.txt')
get_large_audio_transcription(plik_rozpoznany_txt_3,'plik_3.txt')
plik_txt='pan-tadeusz.txt'
procent_1=dokladnosc_rozpoznawania_mowy(plik_txt,'plik_1.txt')
procent_2=dokladnosc_rozpoznawania_mowy(plik_txt,'plik_2.txt')
procent_3=dokladnosc_rozpoznawania_mowy(plik_txt,'plik_3.txt')
procenty_tab=[]
procenty_tab.append(procent_1)
procenty_tab.append(procent_2)
procenty_tab.append(procent_3)
print(procenty_tab)
plt.bar(['Plik bez szumu','Plik -50dB','Plik -30dB z szumem '],procenty_tab)
plt.title('Poprawność rozpoznawania mowy')
plt.xlabel('Kolejne pliki dźwiękowe')
plt.ylabel('Skuteczność rozpoznania')
plt.show()