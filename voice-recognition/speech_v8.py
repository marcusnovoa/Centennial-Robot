""" V8 includes no facial recognition due to COVID-19 and masks.

    We've implemented our own module for Google text-to-speech
    in order to provide voice configuration and receive custom
    voice output.
"""
try:
    from dotenv import load_dotenv # pip3 install python-dotenv
    from ibm_watson import AssistantV2 # pip3 install ibm-watson
    from ibm_cloud_sdk_core.authenticators import IAMAuthenticator # This should install with ibm-watson (see above line)
    import os
    from playsound import playsound # pip3 install playsound
    import speech_recognition as sr # pip3 install SpeechRecognition
    from text_to_speech import gtts_custom
except Exception as e:
    print('[Speech V8] Import Error: {}'.format(e))

load_dotenv() # Load the environment variables from .env file
file_name = 'resources/response.mp3'
sound_ding_path = 'resources/ding.wav'
sound_dong_path = 'resources/dong.wav'

def releaseResponse_Normal(response_text=""):
    # Use text-to-speech to play back audio response
    print('Response:', response_text)
    gtts_custom(response_text, file_name)
    playsound(file_name)

# Authenticate the assistant via authID
authenticator = IAMAuthenticator(os.getenv('WATSON_API_KEY'))
assistant = AssistantV2(
    version='2019-11-19',
    authenticator=authenticator
)

# Set the API location url and the assistant's ID
assistant.set_service_url(os.getenv('WATSON_SERVICE_URL'))
assistant_id = os.getenv('WATSON_ASSISTANT_ID')

# Create the assistant's session (new session is generated each time the code is run)
new_session = assistant.create_session(assistant_id)
session = new_session.get_result()
session_id = session['session_id']

# Getting here means the session was successful and therefore we need to ask the user if they are ready to begin
# input('Press Enter when ready...')

# Initialize the speech recognizer and change threshold
r = sr.Recognizer()
r.dynamic_energy_threshold = False
r.energy_threshold = 700

# BEGIN MAIN LOOP
quit_check = False # Note that there is currently no exit condition for this. The program has to be manually stopped.
while quit_check is not True:

    # Begin probing the mic
    with sr.Microphone() as source:
        print("Listening for 'Hey, Charlie'...")
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source) # This line means it is listening
        # Needs a few try/except - See the excepts for more info
        try:
            text = r.recognize_google(audio) # This line converts what you say into a string
            print("You said: {}".format(text))

            # If what you said was ONLY "hey charlie", then it will look AGAIN for a follow-up line
            if text.lower() == "hey charlie" or text.lower() == "charlie":
                playsound(sound_ding_path) # Sound is played as a signal for "listening"
                audio = r.listen(source)   # Listening for a follow-up sentence

                try:
                    text = r.recognize_google(audio)  # Recognize the follow-up sentence
                    print("You said: {}".format(text))

                    # Send a request to the IBM assistant with the given intent
                    response = assistant.message(
                        assistant_id=assistant_id,
                        session_id=session_id,
                        input={
                            'message_type': 'text',
                            'text': text # This is the text that is FED INTO THE ASSISTANT
                        }
                    ).get_result() # This is the text that is SPIT OUT OF THE ASSISTANT (as json)

                    # Parse out the response from the json object
                    pulled_list = response["output"]["generic"]
                    response_text = pulled_list[0].get('text')

                    releaseResponse_Normal(response_text) # Convert Watson response from text to speech

                except sr.UnknownValueError:
                    print("Could not recognize what you said")

            # If the sentence STARTS WITH "hey charlie" that means there is already an attached sentence
            elif (text.lower()).startswith("hey charlie") or (text.lower()).startswith("charlie"):
                parsed_text = text[12:]  # Get the string without the "hey charlie" at the beginning
                print("PARSED: ", parsed_text)

                # Send a request to the IBM assistant with the given intent
                response = assistant.message(
                    assistant_id=assistant_id,
                    session_id=session_id,
                    input={
                        'message_type': 'text',
                        'text': parsed_text # This is the text that is FED INTO THE ASSISTANT
                    }
                ).get_result() # This is the text that is SPIT OUT OF THE ASSISTANT (as json)

                # Parse out the response from the json object
                pulled_list = response["output"]["generic"]
                response_text = pulled_list[0].get('text')

                releaseResponse_Normal(response_text) # Convert Watson response from text to speech

        except sr.UnknownValueError:
            print("Could not recognize what you said")
