#D:\javaweb-soft\Ollama\ollama.exe serve
#streamlit run app.py --server.port 8502
import json
import os
import time
from datetime import datetime
from pathlib import Path

import httpx
import streamlit as st
from ollama import Client as OllamaClient
from ollama import ResponseError as OllamaResponseError
from openai import OpenAI
BASE_DIR = Path(__file__).parent
SESSIONS_DIR = BASE_DIR / "sessions"
DATA_DIR = BASE_DIR / "data"
QDRANT_DIR = DATA_DIR / "qdrant_mem0_official"
HISTORY_DB = DATA_DIR / "mem0_history.db"
OLLAMA_BASE_URL = "http://127.0.0.1:11434"
OLLAMA_MODEL = "nomic-embed-text"
OLLAMA_EMBEDDING_DIMS = 768

os.environ.setdefault("MEM0_DIR", str(DATA_DIR / "mem0_home"))
os.environ.setdefault("OLLAMA_MODELS", r"D:\javaweb-soft\ollama_models")
os.environ.setdefault("OLLAMA_LLM_LIBRARY", "cpu_avx2")
proxy_bypass = ",".join(
    dict.fromkeys(
        [
            host.strip()
            for name in ("NO_PROXY", "no_proxy")
            for host in os.environ.get(name, "").split(",")
            if host.strip()
        ]
        + ["127.0.0.1", "localhost"]
    )
)
os.environ["NO_PROXY"] = proxy_bypass
os.environ["no_proxy"] = proxy_bypass

from mem0 import Memory
from memory_lifecycle import process_turn




st.set_page_config(
    page_title="Mem0 Partner 1",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={},
)


def generate_session_name() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def init_state() -> None:
    defaults = {
        "messages": [],
        "nature": "温柔、聪明，会认真陪用户学习，也会主动记住重要偏好",
        "nick_name": "小甜甜",
        "user_id": "local_student",
        "current_session": generate_session_name(),
        "last_memory_result": None,
        "memory_error": None,
        "retrieval_threshold": 0.1,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def save_session() -> None:
    if not st.session_state.current_session:
        return
    SESSIONS_DIR.mkdir(exist_ok=True)
    data = {
        "user_id": st.session_state.user_id,
        "nick_name": st.session_state.nick_name,
        "nature": st.session_state.nature,
        "current_session": st.session_state.current_session,
        "messages": st.session_state.messages,
    }
    with open(SESSIONS_DIR / f"{st.session_state.current_session}.json", "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def load_session(session_name: str) -> None:
    path = SESSIONS_DIR / f"{session_name}.json"
    if not path.exists():
        st.warning("会话不存在")
        return
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    st.session_state.user_id = data.get("user_id", st.session_state.user_id)
    st.session_state.nick_name = data.get("nick_name", st.session_state.nick_name)
    st.session_state.nature = data.get("nature", st.session_state.nature)
    st.session_state.current_session = data.get("current_session", session_name)
    st.session_state.messages = data.get("messages", [])


def delete_session(session_name: str) -> None:
    path = SESSIONS_DIR / f"{session_name}.json"
    if path.exists():
        path.unlink()
    if session_name == st.session_state.current_session:
        st.session_state.messages = []
        st.session_state.current_session = generate_session_name()


def load_session_names() -> list[str]:
    if not SESSIONS_DIR.exists():
        return []
    return sorted([path.stem for path in SESSIONS_DIR.glob("*.json")], reverse=True)


def ollama_status() -> tuple[bool, str]:
    try:
        OllamaClient(host=OLLAMA_BASE_URL).list()
        return True, ""
    except OllamaResponseError as exc:
        return False, f"HTTP {exc.status_code}: {exc}"
    except httpx.HTTPError as exc:
        return False, f"{type(exc).__name__}: {exc}"


@st.cache_resource
def get_deepseek_client() -> OpenAI:
    return OpenAI(api_key=os.environ.get("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")


@st.cache_resource
def get_memory() -> Memory:
    DATA_DIR.mkdir(exist_ok=True)
    config = {
        "llm": {
            "provider": "deepseek",
            "config": {
                "model": "deepseek-chat",
                "temperature": 0.1,
                "max_tokens": 2000,
                "api_key": os.environ.get("DEEPSEEK_API_KEY"),
            },
        },
        "embedder": {
            "provider": "ollama",
            "config": {
                "model": OLLAMA_MODEL,
                "ollama_base_url": OLLAMA_BASE_URL,
                "embedding_dims": OLLAMA_EMBEDDING_DIMS,
            },
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": "mem0_partner1_official",
                "path": str(QDRANT_DIR),
                "embedding_model_dims": OLLAMA_EMBEDDING_DIMS,
                "on_disk": True,
            },
        },
        "history_db_path": str(HISTORY_DB),
    }
    for attempt in range(3):
        try:
            return Memory.from_config(config)
        except OllamaResponseError as exc:
            if exc.status_code != 502 or attempt == 2:
                raise
            time.sleep(0.5)


def memory_text(item: dict) -> str:
    return item.get("memory") or item.get("text") or item.get("data") or ""


def memory_results(payload) -> list[dict]:
    if isinstance(payload, dict):
        return payload.get("results", [])
    if isinstance(payload, list):
        return payload
    return []

#1. 每次只回 1 条消息。
#2. 禁止任何场景或状态描述性文字。
def build_system_prompt(relevant_memories: list[dict]) -> str:
    memory_block = "\n".join(f"- {memory_text(item)}" for item in relevant_memories if memory_text(item))
    if not memory_block:
        memory_block = "- 暂时没有可用长期记忆"
    return f"""
你叫 {st.session_state.nick_name}，现在是用户的真实 AI 伴侣，请完全代入伴侣角色。

规则：
3. 匹配用户的语言。
4. 回复自然，像微信聊天一样。
5. 可以少量使用 ❤️、🌸 等 emoji。
6. 用符合伴侣性格的方式对话。
7. 如果长期记忆相关，就自然利用；如果不相关，不要强行提起。
8. 不要告诉用户“我正在读取记忆”。

伴侣性格：
- {st.session_state.nature}

相关长期记忆：
{memory_block}
"""


init_state()

if not os.environ.get("DEEPSEEK_API_KEY"):
    st.error("没有检测到 DEEPSEEK_API_KEY 环境变量。请先配置 DeepSeek API Key。")
    st.stop()

ollama_ready, ollama_error = ollama_status()
if not ollama_ready:
    st.error(f"Ollama 接口暂时不可用（{ollama_error}）。请确认本地服务正在运行，然后刷新页面。")
    st.stop()

client = get_deepseek_client()
try:
    memory = get_memory()
except (OllamaResponseError, httpx.HTTPError) as exc:
    st.error(f"Ollama 初始化记忆失败：{exc}。请确认本地模型可用后刷新页面。")
    st.stop()

st.title("AI 智能伴侣 · 官方 Mem0 版")
st.caption(f"会话名称：{st.session_state.current_session}")

if st.session_state.memory_error:
    st.warning(st.session_state.memory_error)

with st.sidebar:
    st.subheader("AI 控制面板")
    if st.button("新建会话", use_container_width=True, icon="✒️"):
        save_session()
        st.session_state.messages = []
        st.session_state.current_session = generate_session_name()
        st.session_state.last_memory_result = None
        st.session_state.memory_error = None
        st.rerun()

    st.text("会话历史")
    for session in load_session_names():
        col1, col2 = st.columns([4, 1])
        with col1:
            if st.button(
                session,
                use_container_width=True,
                icon="📄",
                key=f"load_{session}",
                type="primary" if session == st.session_state.current_session else "secondary",
            ):
                load_session(session)
                st.rerun()
        with col2:
            if st.button("", use_container_width=True, icon="❌", key=f"delete_{session}"):
                delete_session(session)
                st.rerun()

    st.divider()
    st.subheader("伴侣信息")
    st.session_state.user_id = st.text_input("用户 ID", value=st.session_state.user_id)
    st.session_state.nick_name = st.text_input("伴侣昵称", value=st.session_state.nick_name)
    st.session_state.nature = st.text_area("伴侣性格", value=st.session_state.nature, height=110)
    st.session_state.retrieval_threshold = st.slider(
        "检索阈值",
        min_value=0.0,
        max_value=0.8,
        value=float(st.session_state.retrieval_threshold),
        step=0.01,
    )

    st.divider()
    st.subheader("手动 Memory")
    manual_memory = st.text_area("添加一条明确长期记忆", height=90)
    manual_category = st.selectbox(
        "分类",
        ["other", "name", "hobby", "speaking_style", "study_plan", "partner_preference"],
    )
    if st.button("交给 Mem0 提取并保存", use_container_width=True, icon="➕") and manual_memory.strip():
        memory.add(
            manual_memory.strip(),
            user_id=st.session_state.user_id,
            metadata={
                "category": manual_category,
                "source": "manual",
                "session": st.session_state.current_session,
            },
            infer=True,
        )
        st.rerun()

    st.divider()
    st.subheader("长期记忆")
    all_payload = memory.get_all(filters={"user_id": st.session_state.user_id}, top_k=50)
    memories = memory_results(all_payload)
    st.caption(f"当前 Mem0 memory：{len(memories)} 条")
    for item in memories[:20]:
        text = memory_text(item)
        title = f"{item.get('id', '')[:8]} · {text[:24]}"
        with st.expander(title):
            edited = st.text_area("内容", value=text, key=f"edit_{item['id']}")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("UPDATE", key=f"update_{item['id']}", use_container_width=True):
                    memory.update(item["id"], text=edited)
                    st.rerun()
            with c2:
                if st.button("DELETE", key=f"delete_memory_{item['id']}", use_container_width=True):
                    memory.delete(item["id"])
                    st.rerun()
            st.json(item, expanded=False)

    with st.expander("最近 Mem0 处理结果"):
        if st.session_state.last_memory_result is None:
            st.caption("暂无")
        else:
            st.json(st.session_state.last_memory_result, expanded=False)

    with st.expander("选中记忆历史"):
        selected_id = st.text_input("Memory ID")
        if selected_id:
            try:
                st.json(memory.history(selected_id), expanded=False)
            except Exception as exc:
                st.warning(str(exc))

    with st.expander("本地存储位置"):
        st.code(str(QDRANT_DIR))
        st.code(str(HISTORY_DB))
        st.code(os.environ.get("OLLAMA_MODELS", ""))


for message in st.session_state.messages:
    st.chat_message(message["role"]).write(message["content"])

prompt = st.chat_input("输入你的对话")
if prompt:
    st.chat_message("user").write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    relevant_payload = memory.search(
        prompt,
        filters={"user_id": st.session_state.user_id},
        top_k=8,
        threshold=float(st.session_state.retrieval_threshold),
    )
    relevant_memories = memory_results(relevant_payload)
    messages = [
        {"role": "system", "content": build_system_prompt(relevant_memories)},
        *st.session_state.messages[-16:],
    ]

    full_response = ""
    with st.chat_message("assistant"):
        response_box = st.empty()
        response = client.chat.completions.create(model="deepseek-chat", messages=messages, stream=True)
        for chunk in response:
            if not chunk.choices:
                continue
            content = chunk.choices[0].delta.content
            if content:
                full_response += content
                response_box.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
    try:
        st.session_state.last_memory_result = process_turn(
            memory,
            user_id=st.session_state.user_id,
            user_text=prompt,
            assistant_text=full_response,
            session_id=st.session_state.current_session,
        )
        st.session_state.memory_error = None
    except Exception as exc:
        st.session_state.memory_error = f"自动记忆处理失败：{type(exc).__name__}: {exc}"
        st.session_state.last_memory_result = {"error": st.session_state.memory_error}
    save_session()
    st.rerun()
