from src.generation.llm import LLM
from src.generation.prompts import build_prompt


class AnswerGenerator:

    def __init__(self):
        self.llm = LLM()

    def answer(self, question, results):
        context = ""
        sources = []

        for item in results:
            doc = item["document"]
            context += doc["chunk_text"] + "\n\n"

        source_item = {
            "source": doc["source"],
            "page": doc["page"],
            "section": doc.get(
                "section",
                "Unknown"
            ),
            "evidence": doc["chunk_text"][:250]
        }


        if source_item not in sources:
            sources.append(source_item)

        prompt = build_prompt(question, context)
        answer = self.llm.generate(prompt)

        return {
            "answer": answer,
            "sources": sources,
        }