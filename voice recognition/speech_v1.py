import pyttsx3
import speech_recognition as sr

input("Press Enter when ready...")

tts = pyttsx3.init()
rate = tts.getProperty("rate")
tts.setProperty("rate", 150)
volume = tts.getProperty("volume")
tts.setProperty("volume", 1)

r = sr.Recognizer()
with sr.Microphone() as source:
    print("Talk into mic now")
    audio = r.listen(source)
    try:
        text = r.recognize_google(audio)
        print("You said: {}".format(text))
        tts.say(format(text))
        tts.runAndWait()
    except:
        print("Could not recognize what you said")

'''
AUDIO_FILE = "testing.wav"

r = sr.Recognizer()
with sr.AudioFile(AUDIO_FILE) as source:
    audio = r.record(source)  # read the entire audio file
    print("Transcription: " + r.recognize_google(audio))
'''
