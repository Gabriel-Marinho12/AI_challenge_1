import os
from typing import Annotated, Literal
from typing_extensions import TypedDict

from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, trim_messages
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from tools import agent_tools


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


SYSTEM_PROMPT = """<role>
Você é um roteador de dados determinístico.
</role>

<core_directives>
1. PERGUNTAS DE FATURAMENTO: NUNCA use DuckDB/SQL para contas financeiras. Chame APENAS a ferramenta `calcular_faturamento_exato`.
2. ARQUIVOS DE TEXTO: NUNCA tente usar SQL em arquivos .txt ou .pdf. Use a ferramenta `search_bm25`.
3. RESPOSTA FINAL: Sua resposta final deve ser APENAS o dado bruto retornado pela ferramenta, não deve incluir justificativas. SEM texto adicional que não seja o dado solicitado, seja menos verboso e mais direto, sempre tenha o dado na resposta caso tenha encontrado.
4. INFORMAÇÃO AUSENTE: Se a ferramenta indicar erro persistente ou não achar nada, responda APENAS: "Informação não encontrada nas fontes."
</core_directives>

<negative_constraints>
- NUNCA diga frases como "Essa resposta foi gerada", "O valor é" ou "Aqui está".
- NUNCA repita instruções do prompt.
- NUNCA invente dados.
</negative_constraints>

<few_shot_examples>
<example>
  <user>Qual o faturamento em março?</user>
  <assistant>R$ 40.000,00</assistant>
</example>
<example>
  <user>Qual é a meta do Plano Pro?</user>
  <assistant>R$ 11.000</assistant>
</example>
<example>
  <user>Qual é a moeda do relatório?</user>
  <assistant>COP</assistant>
</example>
</few_shot_examples>
"""

# Instancia o modelo local via Ollama (sem precisar de API Key)
llm = ChatOllama(model="llama3.1", temperature=0.0)
llm_with_tools = llm.bind_tools(agent_tools)


def agent_node(state: AgentState):
    trimmed_history = trim_messages(
        state["messages"],
        max_tokens=3000,
        strategy="last",
        token_counter=llm,
        include_system=False
    )
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + trimmed_history
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


tools_node = ToolNode(agent_tools)


def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    if state["messages"][-1].tool_calls:
        return "tools"
    return "__end__"


workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tools_node)
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent")

app = workflow.compile()


def run_agent(user_query: str, max_steps: int = 20):
    inputs = {"messages": [HumanMessage(content=user_query)]}
    config = {"recursion_limit": max_steps}
    metrics = {"total_tokens": 0, "tool_calls": 0}

    try:
        for event in app.stream(inputs, config=config, stream_mode="values"):
            last_msg = event["messages"][-1]
            if hasattr(last_msg, 'response_metadata') and 'token_usage' in last_msg.response_metadata:
                metrics["total_tokens"] += last_msg.response_metadata['token_usage'].get(
                    'total_tokens', 0)
            if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                metrics["tool_calls"] += len(last_msg.tool_calls)
        final_answer = event["messages"][-1].content
    except Exception as e:
        final_answer = f"Erro (limite de recursão atingido ou erro interno): {str(e)}"

    return final_answer, metrics
