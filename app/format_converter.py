def convert(file, file_or_text, from_type, to_type):
    import pypandoc

    pypandoc.download_pandoc()

    if file_or_text == "text":
        return pypandoc.convert_text(file, to_type, format=from_type)
    else:
        return pypandoc.convert_file(file, to_type, format=from_type)