from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from agent_orchestration.retrieval import delete_document, search_documents


class SearchDocumentsInput(BaseModel):
    query: str = Field(min_length=1, max_length=20_000)
    top_k: int = Field(default=5, ge=1, le=20)


class DeleteDocumentInput(BaseModel):
    document_id: str = Field(min_length=1, max_length=500)


def build_tools() -> list[StructuredTool]:
    return [
        StructuredTool.from_function(
            func=lambda query, top_k=5: [
                result.model_dump()
                for result in search_documents(query, top_k)
            ],
            name="search_internal_documents",
            description="Search internal documents using hybrid keyword and semantic retrieval.",
            args_schema=SearchDocumentsInput,
        ),
        StructuredTool.from_function(
            func=lambda document_id: {
                "document_id": document_id,
                "deleted": delete_document(document_id),
            },
            name="delete_internal_document",
            description="Delete an internal document by ID. Requires human approval.",
            args_schema=DeleteDocumentInput,
        ),
    ]


TOOL_RISK: dict[str, str] = {
    "search_internal_documents": "low",
    "delete_internal_document": "high",
}
