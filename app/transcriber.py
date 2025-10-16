def transcribe(file_in, model = None):

    import whisper

    trans_model = whisper.load_model("small") if model == None else whisper.load_model(model)

    print(f"Transcribing file {file_in}")

    file = file_in

    result = trans_model.transcribe(f"./app/{file}")

    print(f"Transcribed text: {result['text'][:50]}")
    return result['text']