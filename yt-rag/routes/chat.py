# routes/chat.py
from fastapi import APIRouter, UploadFile, File, Query, HTTPException
from services import vector_store, transcript
from typing import Annotated, List
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage, AIMessage
from typing_extensions import TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
import os

router = APIRouter()

# ---------- Configure Gemini ----------
gemini_model = ChatGoogleGenerativeAI(
    model=os.getenv("GEMINI_MODEL"),
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=float(os.getenv("GEMINI_TEMPERATURE", "0.7")),
)


# ---------- State definition ----------
class State(TypedDict):
    messages: Annotated[list, add_messages]  # full conversation
    transcript_context: List[str]  # transcript chunks


# ---------- Memory ----------
memory = InMemorySaver()


# ---------- Helpers ----------
def merge_contexts(
    existing: List[str], new_chunks: List[str], max_chunks: int = 80
) -> List[str]:
    """
    Merge two lists of transcript chunks:
    - Deduplicate
    - Preserve order
    - Keep only the most recent `max_chunks`
    """
    if not existing:
        combined = new_chunks.copy()
    else:
        combined = existing.copy()
        for chunk in new_chunks:
            if chunk not in combined:
                combined.append(chunk)

    if len(combined) > max_chunks:
        combined = combined[-max_chunks:]
    return combined


# ---------- Build LangGraph ----------
def create_graph():
    graph_builder = StateGraph(State)

    def chatbot(state: State):
        if not state["messages"]:
            return {"messages": []}

        question = state["messages"][-1].content
        transcript_chunks = state.get("transcript_context", []) or []
        context_string = "\n\n".join(transcript_chunks[-40:])  # last 40 chunks
        history_msgs = state["messages"][:-1]
        history_string = (
            "\n".join([f"{m.type}: {m.content}" for m in history_msgs])
            if history_msgs
            else ""
        )

        prompt = f"""
                You are a helpful assistant. Use the conversation history and transcript context
                to answer the question as accurately and concisely as possible.
                If the answer is not present in the provided context or history, reply exactly:
                "Answer is not available in the context or history."

                Conversation History:
                {history_string}

                Transcript Context (most relevant chunks):
                {context_string}

                Question:
                {question}

                Answer:
                """

        ai_resp_obj = gemini_model.invoke(prompt)
        ai_text = (
            ai_resp_obj.content if hasattr(ai_resp_obj, "content") else str(ai_resp_obj)
        )
        return {"messages": state["messages"] + [AIMessage(content=ai_text)]}

    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_edge("chatbot", END)
    return graph_builder.compile(checkpointer=memory)


graph = create_graph()

# ---------- Fetch transcript automatically ----------
@router.post("/chat/fetch")
async def fetch_and_index_transcript(
    video_url: str = Query(..., description="YouTube video URL"),
    session_id: str = Query(..., description="Unique session ID"),
    max_chunks: int = Query(80, description="Maximum transcript chunks to retain"),
    lang: str = Query("en", description="Transcript language"),
):
    try:
        text = transcript.fetch_transcript(video_url, lang=lang)
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Transcript fetch failed: {str(e)}"
        )

    new_chunks = vector_store.add_text(text)
    config = {"configurable": {"thread_id": session_id}}

    try:
        existing_state = graph.get_state(config).values
        existing_chunks = existing_state.get("transcript_context", []) or []
        existing_messages = existing_state.get("messages", []) or []
    except Exception:
        existing_chunks, existing_messages = [], []

    merged = merge_contexts(existing_chunks, new_chunks, max_chunks=max_chunks)
    new_state = {"messages": existing_messages, "transcript_context": merged}
    graph.update_state(config, new_state)

    return {
        "indexed_chunks": len(new_chunks),
        "session_id": session_id,
        "video_url": video_url,
    }


# ---------- Query endpoint ----------
@router.get("/chat/query")
async def query_transcript(
    question: str = Query(..., description="Question about the transcript"),
    session_id: str = Query(..., description="Unique session ID"),
    k: int = Query(8, description="Number of vector search results to retrieve"),
):
    retrieved = vector_store.query(question, k=k)
    if not retrieved:
        return {"error": "No relevant context found.", "session_id": session_id}

    config = {"configurable": {"thread_id": session_id}}
    try:
        current_state = graph.get_state(config).values
        existing_chunks = current_state.get("transcript_context", []) or []
        existing_messages = current_state.get("messages", []) or []
    except Exception:
        existing_chunks, existing_messages = [], []

    merged_chunks = merge_contexts(existing_chunks, retrieved, max_chunks=80)
    new_messages = existing_messages + [HumanMessage(content=question)]
    graph.update_state(
        config,
        {"messages": new_messages, "transcript_context": merged_chunks},
        as_node="chatbot",
    )

    response_text = None
    for event in graph.stream(
        {"messages": [HumanMessage(content=question)]}, config=config
    ):
        for value in event.values():
            response_text = value["messages"][-1].content

    if response_text is None:
        raise HTTPException(status_code=500, detail="No response from Gemini")

    return {"gemini_answer": response_text, "session_id": session_id}
