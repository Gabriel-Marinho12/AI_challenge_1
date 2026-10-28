import json
import os
import re
import unicodedata
from app import run_agent


def normalizar(texto):
    if not texto:
        return ""
    texto = unicodedata.normalize('NFD', str(texto)).encode(
        'ascii', 'ignore').decode('utf-8')
    return re.sub(r'[^\w\s]', '', texto).lower().strip()


def juiz_deterministico(resposta_agente, gabarito):
    resp_norm = normalizar(resposta_agente)
    gab_norm = normalizar(gabarito)

    # Regra 1: Informação não encontrada
    if "nao encontrada" in gab_norm:
        if any(p in resp_norm for p in ["nao encontrada", "nao consta", "nao existe", "ausente"]):
            return 1, "Exato (Adversarial)"
        return 0, "Falhou (Adversarial)"

    # Regra 2: Valores Numéricos (Extrai apenas os números para comparar, ignorando R$, pontos e vírgulas)
    numeros_gab = re.sub(r'\D', '', gabarito)
    numeros_resp = re.sub(r'\D', '', resposta_agente)

    if numeros_gab and numeros_gab in numeros_resp:
        return 1, f"Exato (Valor {numeros_gab} contido na resposta)"

    # Regra 3: Textos simples (ex: BRL)
    if gab_norm in resp_norm:
        return 1, "Exato (Texto contido na resposta)"

    return 0, f"Falhou. Resposta do agente não contém o gabarito."


def run_evals(dataset_path="eval_dataset.json"):
    if not os.path.exists(dataset_path):
        print(f"Erro: O arquivo '{dataset_path}' não foi encontrado.")
        return

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    resultados = []
    total_score = total_tools = 0

    print("Iniciando Pipeline de Avaliação DETERMINÍSTICA...")
    for item in dataset:
        print(f"Testando ID {item['id']} ({item['tipo']})...")

        resposta, metricas = run_agent(item["pergunta"])
        nota, raciocinio = juiz_deterministico(
            resposta, item["gabarito_esperado"])

        total_score += nota
        total_tools += metricas.get("tool_calls", 0)

        resultados.append({
            "id": item["id"],
            "tipo": item["tipo"],
            "nota": nota,
            "tool_calls": metricas.get("tool_calls", 0),
            "resposta_agente": resposta,
            "raciocinio_juiz": raciocinio
        })

    num = len(dataset)
    print("\n" + "="*45)
    print("RELATÓRIO FINAL DE ERO/ACERTO (OLLAMA LOCAL)")
    print("="*45)
    print(f"Precisão Global (Accuracy): {(total_score / num) * 100:.2f}%")
    print(
        f"Média de Tool Calls por Pergunta: {total_tools / num if num > 0 else 0:.1f}")
    print("="*45)

    with open("resultados_eval.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    run_evals()
