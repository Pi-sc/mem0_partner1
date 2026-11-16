"""Pass conversation turns to the official Mem0 SDK inference pipeline."""


def process_turn(
    memory,
    *,
    user_id: str,
    user_text: str,
    assistant_text: str,
    session_id: str,
) -> dict:
    """Let the SDK extract and persist memories from both sides of a turn."""
    return memory.add(
        [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ],
        user_id=user_id,
        metadata={"source": "chat", "session": session_id},
        infer=True,
    )
