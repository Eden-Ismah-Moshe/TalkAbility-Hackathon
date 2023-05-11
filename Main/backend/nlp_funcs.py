import os
import openai
import json
from pathlib import Path
import nltk
from googletrans import Translator

class library:
    def __init__(self):
        self.translator = Translator()
        openai.api_key = "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"
        self.model = openai.model = "gpt-3.5-turbo"

    def get_transcript(self):
        path = Path(__file__).parents[2]
        path = os.path.join(path, "call_samples", "sample1.mp3")
        audio_file = open("../../call_samples/sample1.mp3", "rb")
        transcript = openai.Audio.transcribe("whisper-1", audio_file)
        return transcript["text"]

    def get_summery(self, transcript):
        completion = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "user", "content": "summarize short this cal: " + transcript}
            ]
        )
        return completion.choices[0].message["content"]


    def translate(self, transcript):
        return self.translator.translate(transcript, dest='en').text

lib = library()
t = lib.get_transcript()
print(type(t))
print(lib.translate(t))
