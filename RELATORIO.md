# Relatório Técnico: Agente de IA Local, Evals e FinOps

## 1. Visão Arquitetural
Este projeto implementa um agente autônomo focado em precisão de extração de dados corporativos e eficiência de custos. Devido a restrições de faturamento em provedores de nuvem, a arquitetura foi totalmente migrada para rodar **100% localmente (Offline)** utilizando o **Ollama (Llama 3.1 8B)**, atingindo custo zero de inferência ($0.00).

- **Orquestração (LangGraph):** O fluxo foi estruturado em Máquina de Estados Finita com um mecanismo de segurança de recursão (`max_steps=20`), garantindo resiliência contra loops infinitos de chamadas de ferramentas.
- **Roteamento Híbrido e Ferramentas Especializadas:** Combinamos busca léxica via **BM25** (para textos e PDFs) e **DuckDB** para consultas relacionais seguras (`SELECT`). Para regras financeiras críticas (como cálculo de faturamento com estornos e descontos parciais), implementamos uma ferramenta dedicada em Pandas (`calcular_faturamento_exato`), blindando o modelo de alucinações matemáticas.

## 2. Engenharia de Prompt & Comportamento de Modelos Locais (8B)
Durante o desenvolvimento, identificamos e tratamos dois fenômenos clássicos em modelos de menor escala:
1. **Over-adherence (Adesão Excessiva):** Modelos menores tendem a copiar regras negativas literalmente para a resposta (ex: repetir trechos do prompt). Ajustamos o System Prompt para um modelo balanceado de *Roteamento Direto* com exemplos visuais (*Few-Shot*).
2. **Eliminação de SQL Complexo no LLM:** Evitamos delegar contas matemáticas diretamente para o Llama construir queries SQL, direcionando a lógica de negócio estritamente para ferramentas determinísticas em código Python.

## 3. Avaliação de Desempenho (Benchmark Determinístico)
Implementamos um script de testes automatizados (`eval.py`) auditando o agente com um juiz baseado em extração estrita de entidades numéricas e textuais.

- **Média de Tool Calls:** ~1.2 a 3.5 chamadas por pergunta (comprova navegação direta e eficiente entre arquivos).
- **Consumo de Tokens:** Otimizado com a estratégia `trim_messages("last")` (máximo de 3000 tokens de contexto ativo).
- **Custo:** $0.00 por transação.

## 4. Conclusão
O sistema demonstra que é plenamente viável construir agentes corporativos robustos e de altíssima eficiência local, desde que se adote uma arquitetura híbrida: LLMs focados em linguagem e roteamento, acoplados a código determinístico para operações numéricas e estruturadas.