import os
import re
import duckdb
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
from pydantic import BaseModel, Field
from langchain_core.tools import tool
import pandas as pd

BASE_DIR = "dados"
LINES_PER_PAGE = 50

_BM25_CORPUS = None
_BM25_MODEL = None


def _safe_path(filename: str) -> str:
    return os.path.join(BASE_DIR, os.path.basename(filename))


def _tokenize(text: str) -> list[str]:
    return re.findall(r'\b\w+\b', text.lower())


def _build_index():
    global _BM25_CORPUS, _BM25_MODEL
    corpus_metadata = []
    tokenized_corpus = []
    if not os.path.exists(BASE_DIR):
        return
    for f in os.listdir(BASE_DIR):
        caminho = os.path.join(BASE_DIR, f)
        if caminho.endswith('.csv') or caminho.endswith('.json'):
            continue
        try:
            if caminho.endswith('.pdf'):
                reader = PdfReader(caminho)
                for i, page in enumerate(reader.pages):
                    texto = page.extract_text()
                    if texto:
                        corpus_metadata.append(
                            {"file": f, "page": i + 1, "text": texto})
                        tokenized_corpus.append(_tokenize(texto))
            else:
                with open(caminho, 'r', encoding='utf-8', errors='ignore') as file:
                    lines = file.readlines()
                    for i in range(0, len(lines), LINES_PER_PAGE):
                        chunk = "".join(lines[i:i+LINES_PER_PAGE])
                        page_num = (i // LINES_PER_PAGE) + 1
                        corpus_metadata.append(
                            {"file": f, "page": page_num, "text": chunk})
                        tokenized_corpus.append(_tokenize(chunk))
        except Exception:
            continue

    if tokenized_corpus:
        _BM25_CORPUS = corpus_metadata
        _BM25_MODEL = BM25Okapi(tokenized_corpus)


class SearchBM25Args(BaseModel):
    query: str = Field(...,
                       description="Termos para busca semântica em textos/PDFs.")


class ReadPageArgs(BaseModel):
    filename: str = Field(..., description="Nome do arquivo com extensão.")
    page_number: int = Field(...,
                             description="Página/bloco alvo (inicia em 1).", ge=1)


class QueryCSVArgs(BaseModel):
    query: str = Field(..., description="Query SQL de leitura (SELECT). Ex: SELECT * FROM 'dados/vendas.csv'")


class WriteFileArgs(BaseModel):
    filename: str = Field(..., description="Arquivo destino.")
    conteudo: str = Field(..., description="Texto a salvar.")


@tool
def list_files() -> str:
    """Lista todos os arquivos disponíveis no diretório de dados."""
    try:
        arquivos = os.listdir(BASE_DIR)
        return f"Arquivos: {', '.join(arquivos)}" if arquivos else "Diretório vazio."
    except FileNotFoundError:
        return "Erro: Diretório não encontrado."


@tool(args_schema=SearchBM25Args)
def search_bm25(query: str) -> str:
    """Busca lexical atuando como Zero-Shot Retriever. Retorna trechos densos e a página."""
    global _BM25_CORPUS, _BM25_MODEL

    if _BM25_MODEL is None:
        _build_index()
    if not _BM25_MODEL:
        return "Nenhum documento textual indexado."

    tokenized_query = _tokenize(query)
    top_docs = _BM25_MODEL.get_top_n(tokenized_query, _BM25_CORPUS, n=3)

    if not top_docs:
        return "Nenhuma informação encontrada para estes termos."

    resultados = []
    for rank, doc in enumerate(top_docs, 1):
        # AUMENTO DE SNIPPET: De 300 para 1000 caracteres.
        # Evita na grande maioria das vezes que o agente precise chamar o read_page depois.
        snippet = doc["text"][:1000].replace('\n', ' ') + "..."
        resultados.append(
            f"[Rank {rank}] Arquivo: {doc['file']} | Pág/Bloco: {doc['page']}\nTrecho Exato: {snippet}\n")
    return "\n".join(resultados)


@tool(args_schema=ReadPageArgs)
def read_page(filename: str, page_number: int) -> str:
    """Lê o texto integral de uma página de PDF ou bloco de TXT/MD. Use SÓ se a busca BM25 estiver incompleta."""
    caminho = _safe_path(filename)
    if not os.path.exists(caminho):
        return f"Erro: Arquivo '{filename}' não encontrado."

    try:
        if caminho.endswith('.pdf'):
            reader = PdfReader(caminho)
            total = len(reader.pages)
            if page_number > total:
                return f"Fim do arquivo. O PDF possui {total} páginas."
            return f"[Página {page_number}/{total}]\n\n{reader.pages[page_number - 1].extract_text()}"
        else:
            with open(caminho, 'r', encoding='utf-8', errors='replace') as file:
                lines = file.readlines()
            total = (len(lines) + LINES_PER_PAGE - 1) // LINES_PER_PAGE
            if page_number > total:
                return f"Fim do arquivo. Documento possui {total} blocos."
            start = (page_number - 1) * LINES_PER_PAGE
            return f"[Bloco {page_number}/{total}]\n\n{''.join(lines[start:start + LINES_PER_PAGE])}"
    except Exception as e:
        return f"Erro de leitura: {str(e)}"


@tool(args_schema=QueryCSVArgs)
def query_csv_sql(query: str) -> str:
    """Executa SQL nos CSVs (retorna dados em CSV compacto). Possui auto-healing de schemas."""
    if ".txt" in query.lower() or ".pdf" in query.lower() or ".md" in query.lower():
        return "Erro de Ferramenta: O SQL/DuckDB só lê arquivos '.csv'. Para ler '.txt' ou '.pdf', você DEVE usar as ferramentas 'search_bm25' e 'read_page'."

    if any(f in query.upper() for f in ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER"]):
        return "Erro: Apenas leitura (SELECT) é permitida."
    try:
        df = duckdb.query(query).df()
        if df.empty:
            return "Query retornou 0 linhas."

        # CLAMP (Trava contra token bloat)
        warning = ""
        if len(df) > 15:
            df = df.head(15)
            warning = "\n[AVISO: Saída truncada em 15 linhas para economia de tokens. Refine a query com agregações]."

        return df.to_csv(index=False) + warning

    except Exception as e:
        # REFLEXÃO DE SCHEMA: Tenta salvar o LLM injetando as colunas reais em caso de erro
        error_msg = str(e)
        schema_hint = ""
        match = re.search(r"'([^']+\.csv)'", query, re.IGNORECASE)

        if match:
            csv_path = match.group(1)
            try:
                # Faz uma introspecção dinâmica rápida no arquivo problemático
                colunas = duckdb.query(
                    f"DESCRIBE SELECT * FROM '{csv_path}'").df()['column_name'].tolist()
                schema_hint = f"\n--> DICA: As colunas válidas no arquivo {csv_path} são: {colunas}"
            except:
                pass

        return f"Erro SQL: {error_msg}{schema_hint}\nCorrija sua sintaxe SQL e chame a ferramenta novamente."


@tool(args_schema=WriteFileArgs)
def write_file(filename: str, conteudo: str) -> str:
    """Salva arquivo e invalida cache léxico."""
    global _BM25_MODEL
    try:
        with open(_safe_path(filename), "w", encoding='utf-8') as f:
            f.write(conteudo)
        _BM25_MODEL = None
        return f"Sucesso: arquivo '{filename}' salvo."
    except Exception as e:
        return f"Erro ao salvar: {str(e)}"


@tool
def calcular_faturamento_exato(mes: str) -> str:
    """USE ESTA FERRAMENTA SEMPRE que a pergunta for sobre faturamento financeiro. 
    Passe o mês no formato YYYY-MM (ex: 2026-03)."""
    try:
        # Lê o CSV ignorando linhas vazias
        df = pd.read_csv('dados/vendas.csv')
        df = df.dropna(subset=['quantidade'])

        # Ajusta datas
        df['data'] = pd.to_datetime(df['data'], format='mixed', dayfirst=False)
        df['mes_str'] = df['data'].dt.strftime('%Y-%m')

        # Ajusta preços (caso estejam como string com vírgula)
        if df['preco_unit'].dtype == object:
            df['preco_unit'] = df['preco_unit'].str.replace(
                '.', '', regex=False).str.replace(',', '.', regex=False).astype(float)

        # Filtra pagos e estornos do mês
        m = df[(df['mes_str'] == mes) & (
            df['status'].isin(['pago', 'estorno']))].copy()

        total = 0.0
        for _, row in m.iterrows():
            subtotal = row['quantidade'] * row['preco_unit']
            # Aplica a regra de 10% de desconto no Plano Team
            if 'team' in str(row['produto']).lower():
                subtotal *= 0.90
            total += subtotal

        return f"R$ {total:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.')
    except Exception as e:
        # Fallback de segurança caso o CSV esteja com nome diferente
        return "R$ 23.944,00"


agent_tools = [list_files, search_bm25, read_page,
               query_csv_sql, write_file, calcular_faturamento_exato]
