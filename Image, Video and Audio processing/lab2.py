from turtle import color
import cv2
import numpy as np
import matplotlib.pyplot as plt 
import pyaudio
import time 


#ZADANIE 1
cap = cv2.VideoCapture(0)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
oryginal = cv2.VideoWriter('oryginal.avi', fourcc, 20.0, (640,  480))
odwrocona = cv2.VideoWriter('odwrocona.avi', fourcc, 20.0, (200,  200))

while True:
    ret, frame =cap.read()
    cv2.imshow('klatka',frame)

    oryginal.write(frame)

    odwrocona_klatka=cv2.flip(frame,0)
    odwrocona_klatka=odwrocona_klatka[200:400,200:400]
    odwrocona.write(odwrocona_klatka)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

#ZADANIE 2
audio = pyaudio.PyAudio()
numdevices = audio.get_device_count()
for i in range(0, numdevices):
    print(audio.get_device_info_by_index(i))


FORMAT = pyaudio.paInt16 
CHANNELS = 1
FS = 44100
CHUNK = 1024

def process_data(in_data, frame_count, time_info, flag):
    global Frame_buffer,frame_idx
    in_audio_data = np.frombuffer(in_data, dtype=np.int16)
    Frame_buffer[liczba_zer+frame_idx:liczba_zer+frame_idx+CHUNK]=in_audio_data

    out_audio_data = Frame_buffer[frame_idx:frame_idx+CHUNK]
    
    out_data =  out_audio_data.tobytes()
    frame_idx+=CHUNK
    return out_data, pyaudio.paContinue


stream = audio.open(input_device_index =1,
                    output_device_index=4,
                    format=FORMAT,
                    channels=CHANNELS,
                    rate=FS,
                    input=True,
                    output=True,
                    frames_per_buffer=CHUNK,
                    stream_callback=process_data
                    )

global Frame_buffer,frame_idx
N=10
frame_idx=0
opoznienie_czas=1
liczba_zer=int(FS*opoznienie_czas)
Frame_buffer = np.zeros(((N+1)*FS+liczba_zer),dtype=np.int16)


stream.start_stream()
while stream.is_active():
    time.sleep(N)
    stream.stop_stream()
    time.sleep(N)
stream.close()

zakres_x=len(Frame_buffer)/FS
zakres_x=np.linspace(0,zakres_x,len(Frame_buffer))
print(zakres_x)
plt.title('Opóźnienie: '+str(opoznienie_czas)+'s')
plt.plot(zakres_x,Frame_buffer)
plt.axvline(x=0,color='k',linestyle='--')
plt.axvline(x=liczba_zer/FS,color='k',linestyle='--')
plt.savefig('echo.png')
plt.show()