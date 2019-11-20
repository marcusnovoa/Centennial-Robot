import pyttsx3
import speech_recognition as sr
import json
from ibm_watson import AssistantV2
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

authenticator = IAMAuthenticator('WATSON ASSISTANT API KEY HERE')
assistant = AssistantV2(
    version='ASSISTANT VERSION DATE HERE',
    authenticator=authenticator
)

assistant.set_service_url('WATSON ASSISTANT SERVICE URL HERE')
assistant_id = 'WATSON ASSISTANT ASSISTANT ID HERE'
session_id = 'WATSON ASSISTANT SESSION ID HERE'

input("Press Enter when ready...")

tts = pyttsx3.init()
rate = tts.getProperty("rate")
tts.setProperty("rate", 150)
volume = tts.getProperty("volume")
tts.setProperty("volume", 1)

quit_check = False

while quit_check is not True:
    r = sr.Recognizer()
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

        except sr.UnknownValueError:
            print("Could not recognize what you said")

