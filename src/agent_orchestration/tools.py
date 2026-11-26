from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from agent_orchestration.retrieval import DocumentStore, get_store


class SearchDocumentsInput(BaseModel):
    query: str = Field(min_length=1, max_length=20_000)
    top_k: int = Field(default=5, ge=1, le=20)


class DeleteDocumentInput(BaseModel):
    document_id: str = Field(min_length=1, max_length=500)


# Any tool missing from this map is treated as high risk by the approval gate.
TOOL_RISK: dict[str, str] = {
    "search_internal_documents": "low",
    "delete_internal_document": "high",
}


def build_tools(store: DocumentStore | None = None) -> list[StructuredTool]:
    """Agent tools. The store is resolved per call so it is created lazily."""

    def resolve() -> DocumentStore:
        return store if store is not None else get_store()

    def search_internal_documents(query: str, top_k: int = 5) -> list[dict[str, object]]:
        return [result.model_dump() for result in resolve().search(query, top_k)]

    def delete_internal_document(document_id: str) -> dict[str, object]:
        return {"document_id": document_id, "deleted": resolve().delete(document_id)}

    return [
        StructuredTool.from_function(
            func=search_internal_documents,
            name="search_internal_documents",
            description="Search internal documents using hybrid keyword and semantic retrieval.",
            args_schema=SearchDocumentsInput,
        ),
        StructuredTool.from_function(
            func=delete_internal_document,
            name="delete_internal_document",
            description="Delete an internal document by ID. Requires human approval.",
            args_schema=DeleteDocumentInput,
        ),
    ]
