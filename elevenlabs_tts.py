from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from elevenlabs import play
import os

load_dotenv()

elevenlabs = ElevenLabs(
  api_key=os.getenv("ELEVENLABS_API_KEY"),
)

def speak_text(text: str):
    audio = elevenlabs.text_to_speech.convert(
        text=text,
        voice_id="6XVxc5pFxXre3breYJhP", # Norwegian accent
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128",
    )
    play(audio)
