import os
import openai
import json
from pathlib import Path

class library:
    def __init__(self):
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


library = library()
transcript_t = library.get_transcript()
print(transcript_t)
print(library.get_summery(transcript_t))

