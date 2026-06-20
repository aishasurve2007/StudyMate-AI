from pypdf import PdfReader
import os


class PDFLoader:

    def __init__(self, file_path):
        self.file_path = file_path


    def load(self):

        reader = PdfReader(self.file_path)

        documents = []


        for page_number, page in enumerate(reader.pages):

            text = page.extract_text()


            documents.append(
                {
                    "page": page_number + 1,
                    "text": text if text else "",
                    "source": os.path.basename(
                        self.file_path
                    )
                }
            )


        return documents