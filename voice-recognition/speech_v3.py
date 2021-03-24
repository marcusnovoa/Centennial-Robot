import pyttsx3
import speech_recognition as sr
# import os
# from google.cloud import texttospeech
import json
from ibm_watson import AssistantV2
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

authenticator = IAMAuthenticator('WATSON ASSISTANT API KEY HERE')
assistant = AssistantV2(
    version='ASSISTANT VERSION DATE HERE',
    authenticator=authenticator
)

assistant.set_service_url('WATSON SERVICE URL HERE')
assistant_id = 'WATSON ASSISTANT ID HERE'
session = assistant.create_session(assistant_id)
session_id = (str(session)[41:77])  # grab the session's ID

input("Press Enter when ready...")

# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "FILE DIR HERE"

# client = texttospeech.TextToSpeechClient()

tts = pyttsx3.init()
rate = tts.getProperty("rate")
tts.setProperty("rate", 150)
volume = tts.getProperty("volume")
tts.setProperty("volume", 1)

quit_check = False

while quit_check is not True:
    r = sr.Recognizer()
    r.dynamic_energy_threshold = False
    r.energy_threshold = 400
    with sr.Microphone() as source:
        print("Talk into mic now")
        audio = r.listen(source)
        try:
            text = r.recognize_google(audio)
            print("You said: {}".format(text))

            response = assistant.message(
                assistant_id=assistant_id,
                session_id=session_id,
                input={
                    'message_type': 'text',
                    'text': text
                }
            ).get_result()

            response_dump = json.dumps(response)  # indent=2
            response_dict = json.loads(response_dump)
            pulled = response_dict["output"]["generic"]
            response_raw = str(pulled[0])
            response_text = response_raw[35:-2]
            print(response_text)

            tts.say(response_text)
            tts.runAndWait()

            """
            synthesis_input = texttospeech.types.SynthesisInput(text=text)

            voice = texttospeech.types.VoiceSelectionParams(
                language_code="en-US",
                name="en-US-Wavenet-D",  # A
                ssml_gender=texttospeech.enums.SsmlVoiceGender.NEUTRAL)  # MALE

            audio_config = texttospeech.types.AudioConfig(
                audio_encoding=texttospeech.enums.AudioEncoding.MP3, pitch=3.00)

            response = client.synthesize_speech(synthesis_input, voice, audio_config)
            """

        except sr.UnknownValueError:
            print("Could not recognize what you said")
