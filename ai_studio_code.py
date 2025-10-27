
#     generate()
from google import genai
from google.genai import types
from dotenv import load_dotenv
import wave
import os 
import pygame

# Set up the wave file to save the output:
def wave_file(filename, pcm, channels=1, rate=24000, sample_width=2):
   with wave.open(filename, "wb") as wf:
      wf.setnchannels(channels)
      wf.setsampwidth(sample_width)
      wf.setframerate(rate)
      wf.writeframes(pcm)

client = genai.Client(api_key="AIzaSyCFWdJPCekccAXgWD2LFTtWMt3yB4sIVsE")

response = client.models.generate_content(
   model="gemini-2.5-flash-preview-tts",
   contents="Say cheerfully: fuck you, just do your job",
   config=types.GenerateContentConfig(
      response_modalities=["AUDIO"],
      speech_config=types.SpeechConfig(
         voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(
               voice_name='Kore',
            )
         )
      ),
   )
)

data = response.candidates[0].content.parts[0].inline_data.data

file_name='out.wav'
wave_file(file_name, data) # Saves the file to current directory



outpath = "./out.wav"

if os.path.exists(outpath):
    print(f"找到檔案：{outpath}，開始播放")
    pygame.mixer.init()
    pygame.mixer.music.load(outpath)
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
else:
    print(f"找不到檔案：{outpath}，無法播放")