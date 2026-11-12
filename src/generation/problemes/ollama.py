"""
Client minimal de l'API Ollama (/api/chat) pour le prototype de génération de problèmes.

- `schema` : sortie contrainte par un schéma JSON (le modèle ne peut produire que
  ce format, plus de JSON mal formé) ;
- `raisonnement` : mode « thinking » (qwen3, gpt-oss…). True/False, ou un niveau
  "low" | "medium" | "high" pour gpt-oss. Si le modèle ne le supporte pas, on
  relance sans, avec un avertissement ;
- `images` : liste d'images PNG (bytes) pour les modèles de vision.
"""

import base64
import json
import logging
import os
import re
import time

import httpx

log = logging.getLogger("testguardai.problemes")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


class ErreurModele(RuntimeError):
    pass


# Le schéma JSON n'autorise après « \ » que les échappements \" \\ \/ \b \f \n \r \t \u.
# Deux façons pour le LaTeX d'un modèle d'être abîmé au décodage :
#  1. commande commençant par b, f, n, r, t écrite avec UNE barre : « \text » est lu
#     comme tabulation + « ext », « \frac » comme saut de page + « rac » ;
#  2. commande interdite (« \int ») : la grammaire FORCE une lettre autorisée après la
#     barre, et le modèle écrit « \tint » (tabulation + « int »).
# On double la barre dans le cas 1, et on retire la lettre forcée dans le cas 2, sans
# toucher aux vrais sauts de ligne (« \nLa suite… »).
_COMMANDES_BFNRT = {
    "text", "textbf", "textrm", "textit", "times", "theta", "tan", "tanh", "tau", "to", "top",
    "tfrac", "frac", "forall", "right", "rightarrow", "rho", "nabla", "neq", "ne", "nu", "not",
    "newline", "nolimits", "beta", "bar", "binom", "big", "bigl", "bigr", "begin", "boxed",
    "bullet", "bmod"}
# Cas 2 : commandes mathématiques usuelles. Exclues : mots français courts (le, in, pi…)
# qui suivent souvent un vrai saut de ligne.
_COMMANDES_FORCEES = {
    "int", "iint", "sum", "prod", "lim", "sqrt", "infty", "cdot", "cdots", "ldots", "dots",
    "alpha", "gamma", "delta", "Delta", "epsilon", "varepsilon", "lambda", "sigma", "omega",
    "phi", "varphi", "partial", "leq", "geq", "subset", "cup", "cap", "mathbb", "mathrm",
    "ln", "log", "exp", "sin", "cos", "arctan", "dfrac", "left", "quad", "displaystyle",
    "approx", "equiv", "sim", "mapsto", "Rightarrow", "iff", "implies", "vec", "overline",
    "limits", "pm", "mid", "lfloor", "rfloor", "langle", "rangle", "emptyset", "mathcal"}
_ECHAPPEMENT_LETTRES = re.compile(r"(?<!\\)\\([bfnrt])([A-Za-z]*)")


def _reparer(m: re.Match) -> str:
    lettre, suite = m.group(1), m.group(2)
    if lettre + suite in _COMMANDES_BFNRT:
        return "\\\\" + lettre + suite          # cas 1 : « \text » -> « \\text »
    if suite in _COMMANDES_FORCEES:
        return "\\\\" + suite                   # cas 2 : « \tint » -> « \\int »
    return m.group(0)                           # vrai échappement (saut de ligne…)


def reparer_latex(contenu: str) -> str:
    """Répare les commandes LaTeX abîmées par l'échappement JSON (voir ci-dessus)."""
    return _ECHAPPEMENT_LETTRES.sub(_reparer, contenu)


def discuter(modele: str, messages: list[dict], *, schema: dict | None = None,
             raisonnement: bool | str | None = None, images: list[bytes] | None = None,
             temperature: float = 0.2, num_ctx: int = 12288, max_tokens: int = 6000,
             timeout: float = 1800) -> dict:
    """Appelle le modèle ; retourne {"contenu", "json" (si schema), "pensee", "duree_s", "tokens"}.

    max_tokens borne la réponse (raisonnement compris) : sans limite, un petit modèle
    contraint par un schéma peut boucler (ajouter des éléments sans fin).
    """
    messages = [dict(m) for m in messages]
    if images:
        messages[-1]["images"] = [base64.b64encode(i).decode() for i in images]
    corps = {"model": modele, "messages": messages, "stream": False,
             "options": {"temperature": temperature, "num_ctx": num_ctx,
                         "num_predict": max_tokens, "repeat_penalty": 1.1}}
    if schema is not None:
        corps["format"] = schema
    if raisonnement not in (None, False):
        corps["think"] = raisonnement

    t0 = time.monotonic()
    r = httpx.post(f"{OLLAMA_BASE_URL}/api/chat", json=corps, timeout=timeout)
    if r.status_code == 400 and "think" in corps and "thinking" in r.text:
        log.warning("%s ne supporte pas le mode raisonnement : relance sans.", modele)
        corps.pop("think")
        r = httpx.post(f"{OLLAMA_BASE_URL}/api/chat", json=corps, timeout=timeout)
    if r.status_code != 200:
        raise ErreurModele(f"Ollama {r.status_code} : {r.text[:300]}")

    data = r.json()
    msg = data.get("message", {})
    sortie = {"contenu": msg.get("content", ""), "pensee": msg.get("thinking", ""),
              "duree_s": round(time.monotonic() - t0, 1),
              "tokens": data.get("eval_count"), "tokens_entree": data.get("prompt_eval_count")}
    if data.get("done_reason") == "length":
        raise ErreurModele(f"réponse tronquée à {max_tokens} tokens ({modele}) : le modèle boucle "
                           f"ou la tâche demande plus de place (augmenter max_tokens)")
    if schema is not None:
        try:
            sortie["json"] = json.loads(reparer_latex(sortie["contenu"]))
        except json.JSONDecodeError as e:
            raise ErreurModele(f"JSON invalide malgré le schéma : {e} — {sortie['contenu'][:200]}")
    return sortie
