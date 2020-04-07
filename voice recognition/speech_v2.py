# import pyttsx3
import speech_recognition as sr
import os
from google.cloud import texttospeech

input("Press Enter when ready...")

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "path to .json file (api key)"
client = texttospeech.TextToSpeechClient()

"""
tts = pyttsx3.init()
rate = tts.getProperty("rate")
tts.setProperty("rate", 150)
volume = tts.getProperty("volume")
tts.setProperty("volume", 1)
"""

quit_check = False

while quit_check is not True:
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Talk into mic now")
        audio = r.listen(source)
        try:
            text = r.recognize_google(audio)
            print("You said: {}".format(text))

            synthesis_input = texttospeech.types.SynthesisInput(text=text)

            voice = texttospeech.types.VoiceSelectionParams(
                language_code="en-US",
                name="en-US-Wavenet-D",  # A
                ssml_gender=texttospeech.enums.SsmlVoiceGender.NEUTRAL)  # MALE

            audio_config = texttospeech.types.AudioConfig(
                audio_encoding=texttospeech.enums.AudioEncoding.MP3, pitch=3.00)

            response = client.synthesize_speech(synthesis_input, voice, audio_config)

        except sr.UnknownValueError:
            print("Could not recognize what you said")
