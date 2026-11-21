from pydantic import BaseModel, Field

class QuestionRequest(BaseModel):
    question: str = Field(
        min_length=3,
        max_length=1000,
        description="Question about the indexed documents"
    )

class DocumentReference(BaseModel):
    source: str
    section: str
    chunk_index: int
    distance: float

class QuestionResponse(BaseModel):
    question: str
    answer: str
    references: list[DocumentReference]
