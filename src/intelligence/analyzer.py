import re


class DocumentAnalyzer:


    def analyze(self, chunks):

        for chunk in chunks:


            text = chunk["chunk_text"]


            # detect headings

            lines = text.split("\n")


            section = "General"


            for line in lines:

                if len(line)<80 and line.strip():

                    section=line.strip()
                    break



            # keyword extraction

            words = re.findall(
                r"\b[A-Za-z]{5,}\b",
                text
            )


            keywords = list(
                set(words[:10])
            )



            chunk["section"]=section

            chunk["keywords"]=keywords



        return chunks