import json

with open('resultados_eval.json', 'r', encoding='utf-8') as f:
    dados = json.load(f)

for d in dados:
    if d['nota'] == 0:
        print(f"\nID: {d['id']} | Tipo: {d['tipo']}")
        print(f"Agente respondeu: {d['resposta_agente']}")
        print(f"Motivo da reprovação: {d['raciocinio_juiz']}")
        print("-" * 50)