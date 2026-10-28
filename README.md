# Agente Analítico RAG com LangGraph & DuckDB

Solução desenvolvida para o desafio de engenharia de agentes, otimizada para máxima precisão, baixo consumo de tokens e resiliência a falhas.

## 🚀 Arquitetura e Decisões Técnicas
- **Orquestração:** Máquina de Estados Finita via **LangGraph** para prevenção de loops infinitos.
- **Prevenção de Alucinação Sintática:** Uso de *Structured Outputs* (`strict=True`) via **Pydantic**.
- **Busca Eficiente (Zero-Shot Retriever):** Indexação léxica em memória com **`rank_bm25`** (substituindo grep e leituras cegas).
- **Processamento Analítico:** Delegação de consultas tabulares para **DuckDB** (SQL) com reflexão automática de schema e clamp de segurança.
- **Otimização de Custos (FinOps):** Histórico gerenciado via `trim_messages` e serialização compacta em CSV.

## ⚙️ Como Executar no Windows

1. **Abra o PowerShell na pasta do projeto e crie o ambiente virtual:**
   ```bash
   python -m venv venv
   .\venv\Scripts\Activate.ps1