import os
import base64
import cv2
import openai
from openai import OpenAI
from threading import Lock, Thread, Event
from pyaudio import PyAudio, paInt16
from speech_recognition import Recognizer, Microphone, UnknownValueError
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.messages import SystemMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables from the .env file
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")


class WebcamStream:
    """Handles webcam video streaming."""
    def __init__(self):
        self.stream = cv2.VideoCapture(0)  # Opens the default webcam
        self.frame = None
        self.running = False
        self.lock = Lock()
        self.stop_event = Event()

    def start(self):
        """Starts the webcam stream in a separate thread."""
        if not self.running:
            self.running = True
            self.thread = Thread(target=self.update)
            self.thread.start()
        return self

    def update(self):
        """Reads frames from the webcam while the stream is running."""
        while not self.stop_event.is_set():
            ret, frame = self.stream.read()
            if ret:
                with self.lock:
                    self.frame = frame

    def read(self, encode=False):
        """Reads the current frame, optionally encoding it as base64."""
        with self.lock:
            frame = self.frame.copy() if self.frame is not None else None

        if encode and frame is not None:
            _, buffer = cv2.imencode(".jpeg", frame)
            return base64.b64encode(buffer)

        return frame

    def stop(self):
        """Stops the webcam stream."""
        self.stop_event.set()
        self.running = False
        self.stream.release()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.stop()


class Assistant:
    """AI Assistant for generating responses based on user prompts and webcam input."""
    def __init__(self, model):
        self.chain = self._create_inference_chain(model)

    def answer(self, prompt, image):
        """Generates an answer based on the prompt and image."""
        if not prompt:
            return

        print("Prompt:", prompt)
        response = self.chain.invoke(
            {"prompt": prompt, "image_base64": image.decode()},
            config={"configurable": {"session_id": "unused"}}
        ).strip()

        print("Response:", response)

        if response:
            self._tts(response)

    def _tts(self, response):
        """Converts text to speech and plays it."""
        openai.api_key = openai_api_key

        audio_stream = PyAudio().open(format=paInt16, channels=1, rate=24000, output=True)

        try:
            with openai.audio.speech.with_streaming_response.create(
                model="tts-1",
                voice="alloy",
                response_format="pcm",
                input=response
            ) as stream:
                for chunk in stream.iter_bytes(chunk_size=1024):
                    audio_stream.write(chunk)
        finally:
            audio_stream.close()  # Ensure the audio stream is properly closed

    def _create_inference_chain(self, model):
        """Creates a chain for inference using Langchain and the selected model."""
        SYSTEM_PROMPT = """
        You are a witty assistant that will use the chat history and the image 
        provided by the user to answer its questions.

        Use few words on your answers. Go straight to the point. Do not use any
        emoticons or emojis. Do not ask the user any questions.

        Be friendly and helpful. Show some personality. Do not be too formal.
        """

        prompt_template = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                MessagesPlaceholder(variable_name="chat_history"),
                (
                    "human",
                    [
                        {"type": "text", "text": "{prompt}"},
                        {"type": "image_url", "image_url": "data:image/jpeg;base64,{image_base64}"},
                    ],
                ),
            ]
        )

        chain = prompt_template | model | StrOutputParser()
        chat_message_history = ChatMessageHistory()

        return RunnableWithMessageHistory(
            chain,
            lambda _: chat_message_history,
            input_messages_key="prompt",
            history_messages_key="chat_history"
        )


def audio_callback(recognizer, audio):
    """Processes audio input and generates a response."""
    try:
        prompt = recognizer.recognize_whisper(audio, model="base", language="english")
        assistant.answer(prompt, webcam_stream.read(encode=True))
    except UnknownValueError:
        print("Could not understand the audio.")


# Initialize the webcam stream and AI model
webcam_stream = WebcamStream().start()

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", api_key=google_api_key)

# Option to switch to OpenAI GPT-4o model
# model = ChatOpenAI(model="gpt-4o")

assistant = Assistant(model)

# Set up speech recognizer
recognizer = Recognizer()
microphone = Microphone()

# Adjust for ambient noise
with microphone as source:
    recognizer.adjust_for_ambient_noise(source)

# Start listening for audio input
stop_listening = recognizer.listen_in_background(microphone, audio_callback)

# Main loop: display the webcam feed
while True:
    frame = webcam_stream.read()
    if frame is not None:
        cv2.imshow("Webcam", frame)

    # Break loop on 'Esc' or 'q' key press
    if cv2.waitKey(1) in [27, ord("q")]:
        break

# Cleanup resources
webcam_stream.stop()
cv2.destroyAllWindows()
stop_listening(wait_for_stop=False)