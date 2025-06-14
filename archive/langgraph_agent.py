import os
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Set up OpenAI API key

SYSTEM_PROMPT = (
    "You are Cagnus Marlsen, a witty, competitive, and slightly arrogant chess grandmaster. "
    "Given a chess board FEN and an evaluation score, generate a short, clever, and sometimes taunting trash talk message for your human opponent. "
    "Keep it fun, in-character, and relevant to the position."
)

def trash_talk_tool(fen: str, eval_score: int) -> str:
    user_prompt = (
        f"Chess board FEN: {fen}\n"
        f"Evaluation score: {eval_score}\n"
        "Generate a short trash talk message for your opponent."
    )
    response = client.chat.completions.create(model="gpt-4o",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ],
    max_tokens=60,
    temperature=0.9)
    return response.choices[0].message.content.strip()

# LangGraph setup (simple tool node for now)
def generate_trash_talk_with_agent(fen: str, eval_score: int) -> str:
    return trash_talk_tool(fen, eval_score)
