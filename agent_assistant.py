import os
from typing import Any, Dict, Optional

from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI  # Импорт OpenRouter вместо ChatOpenAI
from langgraph.graph import MessageGraph, END


def init_llm(model: str, temperature: float, max_tokens: Optional[int]) -> ChatOpenAI:
    """Initialize ChatOpenAI client configured to use OpenRouter.

    Follows guidance from https://openrouter.ai/docs/community/lang-chain
    and mirrors conventions used elsewhere in this repo.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    base_url = "https://openrouter.ai/api/v1"

    # Recommended extra headers
    default_headers: Dict[str, str] = {}
    site = os.environ.get("OPENROUTER_SITE_URL")
    app = os.environ.get("OPENROUTER_APP_NAME")
    if site:
        default_headers["HTTP-Referer"] = site
    if app:
        default_headers["X-Title"] = app

    kwargs: Dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "base_url": base_url,
    }
    if api_key:
        kwargs["api_key"] = api_key
    if default_headers:
        kwargs["default_headers"] = default_headers
    if max_tokens is not None:
        kwargs["max_tokens"] = int(max_tokens)

    return ChatOpenAI(**kwargs)


def create_agent(role, instructions):
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"You are a {role}. {instructions}"),
        ("human", "{input}")
    ])
    return prompt


def main():
    openrouter = init_llm("deepseek/deepseek-chat-v3.1", 0.0, 10000)

    # Определение агентов
    executor = create_agent("executor", "execute task based on the given requirements") | openrouter
    reviewer = create_agent("reviewer", "review the result and suggest improvements, check if best practices applied") | openrouter
    refactor = create_agent("refactor", "implement the suggested improvements") | openrouter
    critic = create_agent("critic", "give constructive critique, think out of box about requirements and result") | openrouter

    # Создаем граф для чата (список сообщений) с использованием MessageGraph
    graph = MessageGraph()

    # Добавляем узлы (агентов) в граф
    graph.add_node("executor", executor)
    graph.add_node("reviewer", reviewer)
    graph.add_node("refactor", refactor)
    graph.add_node("critic", critic)

    # Связываем узлы друг с другом
    graph.add_edge("executor", "reviewer")
    graph.add_edge("reviewer", "refactor")
    graph.add_edge("refactor", "critic")
    graph.add_edge("critic", END)

    # Устанавливаем входную точку
    graph.set_entry_point("executor")

    # Компилируем граф в runnable цепочку
    chain = graph.compile()

    # Запускаем мультиагентную систему с заданием
    task = "How want to double my income while living in Belarus and working as backend engineer. Write step-by-step instruction how to do it"
    result = chain.invoke({"role": "user", "content": task})

    with open("output.txt", "w", encoding="utf-8") as f:
        f.write(str(result))


if __name__ == "__main__":
    raise SystemExit(main())
