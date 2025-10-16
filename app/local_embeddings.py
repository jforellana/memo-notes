def embed_text(text, section):
    from txtai import Embeddings
    from docling.document_converter import DocumentConverter
    from docling.chunking import HybridChunker
    import os, tarfile
    from io import StringIO

    def textract(section, text):
        chunker = HybridChunker()

        doc = DocumentConverter().convert(source=StringIO(text)).document

        chunks = chunker.chunk(dl_doc=doc)

        for i, c in enumerate(chunks):
            yield {"Section": section, "Text": chunker.contextualize(chunk=c)}


    def embed(section, text):
        embedding_path = "embeddings"
        os.makedirs(f"./{embedding_path}", exist_ok=True)
        if "my_embeddings.tar.gz" in os.listdir(f"./{embedding_path}"):
            with tarfile.open(f"./{embedding_path}/my_embeddings.tar.gz", "r:gz") as tar:
                tar.extractall(path=f"./{embedding_path}/tmp")

            with Embeddings().load(f"./{embedding_path}/tmp") as embeddings:
                embeddings.upsert(textract(section, text))
                print("embeddings complete!")

                embeddings.save(f"./{embedding_path}/my_embeddings.tar.gz")  
        else:
            with Embeddings() as embeddings:
                embeddings.upsert(textract(section, text))
                print("embeddings complete!")

                embeddings.save(f"./{embedding_path}/my_embeddings.tar.gz")

    print(f"embeddings texts for section {section}")
    embed(section, text)



   


