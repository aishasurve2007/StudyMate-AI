import uuid
from datetime import datetime


def add_metadata(documents):

    for doc in documents:

        doc["document_id"] = str(uuid.uuid4())

        doc["created_at"] = datetime.now().isoformat()

    return documents