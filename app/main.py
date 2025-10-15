from transcriber import transcribe
from summarizer import summarize
from dotenv import load_dotenv
import os

# load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

# PROMPT = """
# Select a number to start or 'quit' to quit:
# 1. Transcribe audio.
# 2. Create transcript summary and highlights, and send to OneNote.
# """

# action=input(PROMPT)

# transcript = ""

# while action != "quit":
#     if action == "1":
#         file = input("Enter the file name (audio.mp3):\t")
#         transcript = transcribe(file)
#     if action == "2":
#         transcript = ""
#         with open("./app/davidClark.txt", "r") as file:
#             transcript += file.read()
#         summary = summarize(transcript, api_key)
#     action = input(PROMPT)

for file in os.listdir("app/audios"):
    transcript = transcribe(f"audios/{file}")
    summarize(transcript, api_key, bulk = True)
    print(f"{file} has been transcribed and saved")