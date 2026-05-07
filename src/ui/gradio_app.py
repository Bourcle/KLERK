import asyncio
import uuid
from collections.abc import AsyncGenerator

import gradio as gr

from service import LegalAgentService
from utils.config import get_settings
from utils.exceptions import AppError
from utils.logging_utils import get_logger
from dotenv import load_dotenv

load_dotenv(override=True)

SETTINGS = get_settings()
SERVICE = LegalAgentService(SETTINGS)
LOGGER = get_logger(__name__)

CUSTOM_CSS = """
.klerk-header {
    background: linear-gradient(135deg, #0f4c75 0%, #1b262c 100%);
    color: white;
    padding: 1.2rem 1.5rem;
    border-radius: 12px;
    margin-bottom: 0.8rem;
    text-align: center;
}
.klerk-header h1 {
    margin: 0 0 0.3rem 0;
    font-size: 1.6rem;
    font-weight: 700;
    letter-spacing: 0.5px;
}
.klerk-header p {
    margin: 0;
    font-size: 0.9rem;
    opacity: 0.85;
}
.pipeline-status {
    font-family: "Menlo", "Consolas", monospace;
    font-size: 0.82rem;
    line-height: 1.5;
    white-space: pre-wrap;
    max-height: 140px;
    overflow-y: auto;
    color: #555;
}
"""


def ensure_session_id(session_id: str | None) -> str:
    """Return an existing session ID or create a new one.

    Args:
        session_id: Existing session ID, or None when a new session should be created.

    Returns:
        str: Existing or newly generated session ID.
    """

    return session_id or uuid.uuid4().hex


def ensure_user_id(user_id: str | None) -> str:
    """Return an existing user ID or the default local user ID.

    Args:
        user_id: Existing user ID, or None when the default user should be used.

    Returns:
        str: Existing user ID or default local user ID.
    """

    return user_id or "local-user"


def normalize_chat(chat) -> list[dict[str, str]]:
    """Normalize Gradio chat history into role-content message dictionaries.

    Args:
        chat: Raw chat history from Gradio or compatible message objects.

    Returns:
        list[dict[str, str]]: Normalized chat messages containing only user and assistant roles.
    """

    if not chat:
        return list()

    normalized: list[dict[str, str]] = list()
    for item in chat:
        if isinstance(item, dict):
            role = item.get("role")
            content = item.get("content", "")
            if role in {"user", "assistant"}:
                normalized.append({"role": role, "content": str(content)})
        elif hasattr(item, "role") and hasattr(item, "content"):
            role = getattr(item, "role", None)
            content = getattr(item, "content", "")
            if role in {"user", "assistant"}:
                normalized.append({"role": role, "content": str(content)})
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            user_text, assistant_text = item
            if user_text:
                normalized.append({"role": "user", "content": str(user_text)})
            if assistant_text:
                normalized.append({"role": "assistant", "content": str(assistant_text)})
    return normalized


def serialize_chat(chat: list[dict[str, str]]) -> list[dict[str, str]]:
    """Serialize chat messages into Gradio-compatible role-content dictionaries.

    Args:
        chat: Normalized chat messages.

    Returns:
        list[dict[str, str]]: Serialized chat messages containing valid user and assistant roles.
    """

    return [
        {"role": item["role"], "content": str(item.get("content", ""))}
        for item in chat
        if item.get("role") in {"user", "assistant"}
    ]


def format_status(session_id: str, phase: str, result=None) -> str:
    """Format the current session, model, and retrieval harness status text.

    Args:
        session_id: Current chat session ID.
        phase: Current UI or pipeline phase.
        result: Optional service result containing route, retrieval, fallback, and MCP metadata.

    Returns:
        str: Human-readable status text for the Gradio sidebar.
    """

    base = (
        f"session_id: {session_id}\n"
        f"phase: {phase}\n"
        f"provider: {SETTINGS.local_llm_provider}\n"
        f"model: {SETTINGS.local_llm_model}\n"
        f"embedding: {SETTINGS.local_embedding_model}"
    )
    if result is not None:
        harness_info = (
            f"\n---\n"
            f"route: {result.route.source_type}/{result.route.topic}\n"
            f"collection: {result.route.collection}\n"
            f"rewritten_query: {result.rewritten_query[:80] if result.rewritten_query else 'N/A'}\n"
            f"docs: {len(result.retrieved_docs)}\n"
            f"iterations: {result.retrieval_iterations}\n"
            f"fallback: {', '.join(result.fallback_history) if result.fallback_history else 'none'}\n"
            f"mcp: {'yes' if result.used_mcp else 'no'}"
        )
        base += harness_info
    return base


def format_result_text(result) -> str:
    """Extract answer text from a service result.

    Args:
        result: Service result object containing the generated answer.

    Returns:
        str: Generated answer text.
    """

    return result.answer


def format_user_error(exc: Exception) -> str:
    """Convert an exception into a user-facing error message.

    Args:
        exc: Exception raised during chat processing.

    Returns:
        str: User-facing error message.
    """

    if isinstance(exc, AppError):
        return exc.user_message
    return "질문 처리 중 알 수 없는 오류가 발생했습니다. 서버 로그를 확인해 주세요."


async def stream_text(
    chat: list[dict[str, str]],
    text: str,
    session_id: str,
    result=None,
) -> AsyncGenerator[tuple[str, list[dict[str, str]], str, str], None]:
    """Stream answer text character by character into the chat state.

    Args:
        chat: Current normalized chat messages.
        text: Full answer text to stream.
        session_id: Current chat session ID.
        result: Optional service result used to format pipeline status.

    Yields:
        tuple[str, list[dict[str, str]], str, str]: Updated session ID, chat messages, status text, and input box value.
    """

    partial = ""
    for ch in text:
        partial += ch
        chat[-1]["content"] = partial
        yield session_id, serialize_chat(chat), format_status(session_id, "streaming", result), ""
        await asyncio.sleep(SETTINGS.char_stream_delay)


async def ui_send(
    session_id: str | None,
    chat: list[dict[str, str]] | None,
    user_text: str,
):
    """Handle a user chat submission and stream the generated answer to the UI.

    Args:
        session_id: Current session ID, or None when a new session ID should be created.
        chat: Current chat history from the Gradio chatbot.
        user_text: User-submitted question text.

    Yields:
        tuple[str, list[dict[str, str]], str, str]: Updated session ID, chat messages, status text, and input box value.
    """

    user_text = (user_text or "").strip()
    chat = normalize_chat(chat)
    session_id = ensure_session_id(session_id)
    user_id = ensure_user_id(None)

    if not user_text:
        yield session_id, serialize_chat(chat), "질문을 입력해 주세요.", ""
        return

    chat.append({"role": "user", "content": user_text})
    chat.append({"role": "assistant", "content": ""})
    yield session_id, serialize_chat(chat), format_status(session_id, "running"), ""

    result = None
    try:
        result = await SERVICE.aask(question=user_text, user_id=user_id, thread_id=session_id)
        answer_text = format_result_text(result)
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception(
            "Chat request failed",
            extra={
                "event": "chat_error",
                "thread_id": session_id,
                "user_id": user_id,
                "error_type": type(exc).__name__,
            },
        )
        answer_text = format_user_error(exc)

    async for payload in stream_text(chat, answer_text, session_id, result):
        yield payload


def ui_new_chat() -> tuple[str, list[dict[str, str]], str]:
    """Create a new chat session with an empty message history.

    Returns:
        tuple[str, list[dict[str, str]], str]: New session ID, empty chat history, and status text.
    """

    session_id = uuid.uuid4().hex
    return session_id, list(), format_status(session_id, "new_chat")


def ui_clear_chat(session_id: str | None) -> tuple[str, list[dict[str, str]], str, str]:
    """Clear the current chat history while preserving or creating a session ID.

    Args:
        session_id: Current session ID, or None when a new session ID should be created.

    Returns:
        tuple[str, list[dict[str, str]], str, str]: Session ID, empty chat history, status text, and cleared input value.
    """

    session_id = ensure_session_id(session_id)
    return session_id, list(), format_status(session_id, "cleared"), ""


def build_app() -> gr.Blocks:
    """Build the Gradio UI for the KLERK chat application.

    Returns:
        gr.Blocks: Configured Gradio Blocks application.
    """

    initial_session_id = uuid.uuid4().hex

    with gr.Blocks(
        title="KLERK",
        css=CUSTOM_CSS,
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="cyan",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("Inter"),
        ),
    ) as demo:
        gr.HTML(
            '<div class="klerk-header">'
            "<h1>KLERK</h1>"
            "<p>Korean Law Engine for Retrieval and Knowledge | Retrieval Harness 기반 법률 QA</p>"
            "</div>"
        )

        session_id = gr.State(value=initial_session_id)

        with gr.Row():
            with gr.Column(scale=1, min_width=300):
                with gr.Group():
                    gr.Markdown("### 채팅 세션")
                    btn_new = gr.Button("새 채팅", variant="primary", size="sm")
                    btn_clear = gr.Button("대화 초기화", variant="secondary", size="sm")

                with gr.Accordion("현재 설정", open=False):
                    gr.Markdown(
                        "\n".join(
                            [
                                f"- LLM Provider: {SETTINGS.local_llm_provider}",
                                f"- LLM Model: {SETTINGS.local_llm_model}",
                                f"- LLM Base URL: {SETTINGS.local_llm_base_url}",
                                f"- Embedding Provider: {SETTINGS.local_embedding_provider}",
                                f"- Embedding Model: {SETTINGS.local_embedding_model}",
                            ]
                        )
                    )

                pipeline_status = gr.Markdown(
                    format_status(initial_session_id, "ready"),
                    elem_classes=["pipeline-status"],
                )

            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    height=520,
                    label="KLERK Chat",
                )
                user_text = gr.Textbox(
                    label="질문 입력",
                    placeholder="예: 민법 제750조의 불법행위 요건을 설명해줘",
                    lines=2,
                )
                with gr.Row():
                    btn_send = gr.Button("전송", variant="primary")
                    btn_clear_input = gr.Button("입력 지우기")

        btn_send.click(
            ui_send,
            inputs=[session_id, chatbot, user_text],
            outputs=[session_id, chatbot, pipeline_status, user_text],
        )
        user_text.submit(
            ui_send,
            inputs=[session_id, chatbot, user_text],
            outputs=[session_id, chatbot, pipeline_status, user_text],
        )
        btn_new.click(
            ui_new_chat,
            inputs=list(),
            outputs=[session_id, chatbot, pipeline_status],
        )
        btn_clear.click(
            ui_clear_chat,
            inputs=[session_id],
            outputs=[session_id, chatbot, pipeline_status, user_text],
        )
        btn_clear_input.click(
            lambda _: "",
            inputs=[user_text],
            outputs=[user_text],
        )

        demo.queue()

    return demo


def launch() -> None:
    """Build and launch the Gradio application server.

    Returns:
        None: This function launches the app and blocks according to Gradio server behavior.
    """

    LOGGER.info(
        "Launching Gradio app",
        extra={
            "event": "gradio_launch",
            "provider": SETTINGS.local_llm_provider,
            "model": SETTINGS.local_llm_model,
            "base_url": SETTINGS.local_llm_base_url,
            "embedding_provider": SETTINGS.local_embedding_provider,
            "embedding_model": SETTINGS.local_embedding_model,
        },
    )
    demo = build_app()
    demo.launch(server_name=SETTINGS.gradio_host, server_port=SETTINGS.gradio_port)
