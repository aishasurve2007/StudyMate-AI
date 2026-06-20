from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentChunker:


    def __init__(self):

        self.splitter = RecursiveCharacterTextSplitter(

            chunk_size=800,

            chunk_overlap=150,

            separators=[
                "\n\n",
                "\n",
                ". ",
                " "
            ]

        )


    def split_documents(self, documents):

        chunks = []


        for doc in documents:


            split_texts = self.splitter.split_text(
                doc["text"]
            )


            for text in split_texts:


                chunks.append(
                    {

                    "chunk_text": text,

                    "page": doc["page"],

                    "source": doc["source"],

                    "document_id":
                    doc["document_id"]

                    }
                )


        return chunks