import os
import re
import json
import logging
import tempfile
import time
import unicodedata

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

load_dotenv()

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
NUM_QUESTIONS = int(os.getenv("NUM_QUESTIONS", "10"))
SECTION_CHAR_SIZE = int(os.getenv("SECTION_CHAR_SIZE", "1800"))
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "8192"))


def _distribute_questions(sections: list, total: int) -> list[int]:
    """Répartit `total` questions sur les sections, pour couvrir tout le document.

    S'il y a plus de sections que de questions à générer (cas attendu avec des
    petites sections), une seule question est attribuée à `total` sections choisies
    uniformément sur le document, les autres reçoivent 0 (pas de minimum de 1 partout,
    sinon la somme dépasserait `total` sans pouvoir redescendre).
    """
    n = len(sections)
    if n == 0:
        return []

    if n >= total:
        counts = [0] * n
        step = n / total
        for i in range(total):
            idx = min(round(i * step), n - 1)
            counts[idx] += 1
        return counts

    lengths = [len(s.page_content) for s in sections]
    total_len = sum(lengths)
    counts = [max(1, round(total * length / total_len)) for length in lengths]

    diff = total - sum(counts)
    i = 0
    while diff != 0:
        idx = i % n
        if diff > 0:
            counts[idx] += 1
            diff -= 1
        elif counts[idx] > 1:
            counts[idx] -= 1
            diff += 1
        i += 1
    return counts


def _escape_stray_backslashes(raw: str) -> str:
    """Le modèle écrit souvent du LaTeX brut (\\frac, \\theta...) sans doubler le
    backslash comme demandé. JSON interprète alors \\f, \\n, \\t... comme des
    caractères de contrôle, corrompant la formule (ex: "\\frac" devient "\\x0crac").
    On double tout backslash isolé, en laissant intactes les paires \\" et \\\\
    déjà correctement échappées (sans quoi on les casserait en les retraitant)."""
    out = []
    i, n = 0, len(raw)
    while i < n:
        if raw[i] == "\\" and i + 1 < n and raw[i + 1] in '"\\':
            out.append(raw[i:i + 2])
            i += 2
        elif raw[i] == "\\":
            out.append("\\\\")
            i += 1
        else:
            out.append(raw[i])
            i += 1
    return "".join(out)


def _clean_pdf_text(text: str) -> str:
    """Supprime les artefacts courants d'extraction PDF (caractères de remplacement, boîtes, etc.)."""
    text = text.replace('�', '')
    text = text.replace('□', '')   # □ WHITE SQUARE
    text = text.replace('■', '')   # ■ BLACK SQUARE
    text = unicodedata.normalize('NFKC', text)
    return text


_TEXT_CMD_RE = re.compile(r'\\text\{(\\[a-zA-Z]+)\}')
_BARE_CMD_RE = re.compile(r'(?<![\\a-zA-Z])(cdot|hbar|times|nabla|partial|infty|forall|exists|langle|rangle)(?![a-zA-Z])')


def _fix_model_questions(questions: list[dict]) -> list[dict]:
    """Corrige les erreurs LaTeX fréquentes générées par le modèle."""
    for q in questions:
        for field in ('reponse', 'raisonnement', 'source'):
            val = q.get(field) or ''
            if not val:
                continue
            # \text{\rangle} → \rangle  (KaTeX refuse les commandes LaTeX dans \text{})
            val = _TEXT_CMD_RE.sub(r'\1', val)
            # `cdot` → `\cdot`, `hbar` → `\hbar`, etc.
            val = _BARE_CMD_RE.sub(r'\\\1', val)
            # Nettoie les caractères de remplacement résiduels
            val = val.replace('□', '').replace('�', '')
            q[field] = val
    return questions


def _build_prompt(filename: str, n_questions: int, text: str) -> str:
    return f"""Tu es un expert pédagogique. À partir du texte suivant extrait du document "{filename}", génère exactement {n_questions} couples question-réponse pertinents pour un examen.

Retourne uniquement un objet JSON valide avec une clé "questions" contenant un tableau, sous ce format exact (respecte cet ordre de champs) :
{{"questions": [{{"question": "...", "raisonnement": "...", "reponse": "...", "source": "..."}}, ...]}}

Règles strictes :
- N'utilise QUE les informations présentes littéralement dans le texte ci-dessous.
- INTERDICTION ABSOLUE d'inventer un exercice, un numéro d'exercice ("Exercice 3a", "4d", etc.) ou un énoncé de calcul qui n'apparaît pas mot pour mot dans le texte. Si un tel numéro ou énoncé n'est pas présent dans le texte, ne le mentionne pas.
- Ne pose des questions QUE sur des définitions, théorèmes ou propriétés explicitement énoncés dans le texte ci-dessous. Pas de question de calcul ou d'exercice à résoudre, sauf si l'énoncé ET toutes les données nécessaires figurent littéralement dans le texte.
- Le champ "source" doit être une CITATION EXACTE, mot pour mot, d'une phrase ou portion de phrase du texte — jamais une reformulation, un résumé ou une paraphrase.
- Si tu ne trouves aucune phrase du texte à citer mot pour mot comme source, n'inclus pas cette question.
- Le champ "raisonnement" : pour une question de calcul ou de démonstration mathématique, déroule ici les étapes intermédiaires. Pour une question non mathématique, laisse une chaîne vide "".
- Le champ "reponse" ne doit contenir QUE la conclusion finale (le résultat, la définition, l'explication), jamais les étapes intermédiaires : celles-ci vont uniquement dans "raisonnement".
- Toute formule ou expression mathématique (dans "raisonnement", "reponse" ET "source") doit être encadrée par $...$ (en ligne) ou $$...$$ (bloc), jamais en texte brut (pas de "x^2" ou "1/2", écris plutôt $x^2$ ou $\\frac{{1}}{{2}}$). Échappe bien chaque backslash LaTeX en JSON (\\\\frac, pas \\frac).
- Règles LaTeX strictes : toujours écrire le backslash devant les commandes ($\\cdot$ jamais $cdot$, $\\hbar$ jamais $hbar$, $\\nabla$ jamais $nabla$, etc.). Ne jamais mettre une commande LaTeX à l'intérieur de \\text{{}} — par exemple $\\langle x \\rangle$ est correct, $\\text{{\\langle}} x \\text{{\\rangle}}$ est interdit.

Les questions doivent tester la compréhension du contenu, pas la mémorisation brute.
Les réponses doivent être complètes et précises.

Texte :
{text}
"""


def ingest_and_generate_questions(file_bytes: bytes, filename: str) -> list[dict]:
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
    finally:
        os.unlink(tmp_path)

    # Sections couvrant tout le document : pas de retrieval par similarité ici,
    # il n'y a pas de question à comparer, le but est la couverture complète.
    for doc in documents:
        doc.page_content = _clean_pdf_text(doc.page_content)

    splitter = RecursiveCharacterTextSplitter(chunk_size=SECTION_CHAR_SIZE, chunk_overlap=0)
    sections = splitter.split_documents(documents)

    llm = ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        format="json",
        temperature=0.7,
        num_ctx=OLLAMA_NUM_CTX,
    )

    question_counts = _distribute_questions(sections, NUM_QUESTIONS)

    all_questions = []
    for section, n_questions in zip(sections, question_counts):
        if n_questions == 0:
            continue
        prompt = _build_prompt(filename, n_questions, section.page_content)

        logger.error(
            "[GEN] Contexte envoyé au modèle pour %s : %d caractères | extrait : %r",
            filename, len(section.page_content), section.page_content[:300],
        )

        t0 = time.monotonic()
        response = llm.invoke([HumanMessage(content=prompt)])
        elapsed = time.monotonic() - t0

        logger.error("[GEN] Appel modèle pour %s : %.1fs", filename, elapsed)
        logger.error("[GEN] Réponse brute du modèle pour %s : %r", filename, response.content)

        try:
            data = json.loads(_escape_stray_backslashes(response.content))
        except json.JSONDecodeError as e:
            logger.error("[GEN] Échec json.loads() pour %s : %r", filename, e)
            continue
        # Accepter {"questions": [...]} ou directement [...]
        section_questions = (
            data.get("questions", list(data.values())[0] if data else [])
            if isinstance(data, dict) else data
        )
        all_questions.extend(_fix_model_questions(section_questions))

    logger.error("[GEN] Liste finale de questions pour %s : %r", filename, all_questions)

    return all_questions
