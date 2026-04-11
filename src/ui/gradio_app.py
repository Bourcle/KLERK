import asyncio
import uuid
from collections.abc import AsyncGenerator

import gradio as gr

from service import LegalAgentService
from utils.config import get_settings
from utils.exceptions import AppError
from utils.logging_utils import get_logger


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
    return session_id or uuid.uuid4().hex


def ensure_user_id(user_id: str | None) -> str:
    return user_id or "local-user"


def normalize_chat(chat) -> list[dict[str, str]]:
    if not chat:
        return []

    normalized: list[dict[str, str]] = []
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
    return [
        {"role": item["role"], "content": str(item.get("content", ""))}
        for item in chat
        if item.get("role") in {"user", "assistant"}
    ]


def format_status(session_id: str, phase: str) -> str:
    return (
        f"session_id: {session_id}\n"
        f"phase: {phase}\n"
        f"provider: {SETTINGS.local_llm_provider}\n"
        f"model: {SETTINGS.local_llm_model}\n"
        f"embedding: {SETTINGS.local_embedding_model}"
    )


def format_result_text(result) -> str:
    return (
        result.answer
        + f"\n\n[route] {result.route.source_type}/{result.route.topic}"
        + f"\n[collection] {result.route.collection}"
        + f"\n[used_mcp] {str(result.used_mcp).lower()}"
    )


def format_user_error(exc: Exception) -> str:
    if isinstance(exc, AppError):
        return exc.user_message
    return "질문 처리 중 알 수 없는 오류가 발생했습니다. 서버 로그를 확인해 주세요."


async def stream_text(
    chat: list[dict[str, str]],
    text: str,
    session_id: str,
) -> AsyncGenerator[tuple[str, list[dict[str, str]], str, str], None]:
    partial = ""
    for ch in text:
        partial += ch
        chat[-1]["content"] = partial
        yield session_id, serialize_chat(chat), format_status(session_id, "streaming"), ""
        await asyncio.sleep(SETTINGS.char_stream_delay)


async def ui_send(
    session_id: str | None,
    chat: list[dict[str, str]] | None,
    user_text: str,
):
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

    async for payload in stream_text(chat, answer_text, session_id):
        yield payload


def ui_new_chat() -> tuple[str, list[dict[str, str]], str]:
    session_id = uuid.uuid4().hex
    return session_id, [], format_status(session_id, "new_chat")


def ui_clear_chat(session_id: str | None) -> tuple[str, list[dict[str, str]], str, str]:
    session_id = ensure_session_id(session_id)
    return session_id, [], format_status(session_id, "cleared"), ""


def build_app() -> gr.Blocks:
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
            "<p>한국 법률 질의응답 | LangGraph Memory | Chroma Retrieval | MCP Fallback</p>"
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
                                f"- LLM Provider: `{SETTINGS.local_llm_provider}`",
                                f"- LLM Model: `{SETTINGS.local_llm_model}`",
                                f"- LLM Base URL: `{SETTINGS.local_llm_base_url}`",
                                f"- Embedding Provider: `{SETTINGS.local_embedding_provider}`",
                                f"- Embedding Model: `{SETTINGS.local_embedding_model}`",
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
            inputs=[],
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
