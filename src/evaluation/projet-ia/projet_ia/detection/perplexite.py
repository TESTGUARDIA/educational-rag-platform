"""
detection/perplexite.py
=======================
Features de PERPLEXITÉ : à quel point un texte est prévisible pour un modèle de
langue causal (GPT-2, BLOOM…), qui prédit chaque token à partir des précédents.

Un LLM génère en choisissant des tokens probables : son texte est en général
peu surprenant (log-probabilité élevée), et régulièrement peu surprenant d'une
phrase à l'autre. Un humain fait des choix plus personnels, avec des « pics de
surprise ». Le modèle de langue n'est PAS entraîné ici : c'est un instrument de
mesure fixe, seul le classifieur qui lit ces mesures apprend.

Le paragraphe entier passe UNE fois dans le modèle (chaque token profite du
contexte des phrases précédentes) ; les tokens sont ensuite répartis entre les
phrases grâce à leur position dans le texte.

Par token : log-probabilité du token réel, entropie de la prédiction (incertitude
du modèle) et « top-1 » (le token réel était-il le choix n°1 du modèle ?).
"""

import contextlib

import numpy as np

NOMS_FEATURES_PPL = [
    "ppl_logprob_moy",           # surprise moyenne (plus haut = plus prévisible)
    "ppl_logprob_std",           # irrégularité de la surprise d'un token à l'autre
    "ppl_top1",                  # part des tokens qui étaient le choix n°1 du modèle
    "ppl_entropie_moy",          # incertitude moyenne du modèle
    "ppl_ratio_entropie",        # surprise / incertitude : corrige les textes prévisibles par nature
    "ppl_std_entre_phrases",     # irrégularité de la surprise moyenne d'une phrase à l'autre
]
N_FEATURES_PPL = len(NOMS_FEATURES_PPL)

NOMS_FEATURES_PPL_PHRASE = ["ppl_logprob_moy", "ppl_logprob_std", "ppl_top1", "ppl_entropie_moy"]
N_FEATURES_PPL_PHRASE = len(NOMS_FEATURES_PPL_PHRASE)


def _stats(logprob, top1, entropie) -> np.ndarray:
    if len(logprob) == 0:
        return np.zeros(N_FEATURES_PPL_PHRASE, dtype=np.float32)
    return np.array([logprob.mean(), logprob.std(), top1.mean(), entropie.mean()],
                    dtype=np.float32)


def bornes_phrases(texte: str, phrases: list[str]) -> np.ndarray:
    """Position (en caractères) du début de chaque phrase dans le texte."""
    debuts, pos = [], 0
    for p in phrases:
        i = texte.find(p, pos)
        if i < 0:                     # phrase introuvable telle quelle : on garde la position courante
            i = pos
        debuts.append(i)
        pos = i + len(p)
    return np.asarray(debuts)


def features_perplexite(texte: str, phrases: list[str], tokens: dict) -> tuple:
    """
    Features de perplexité d'un paragraphe et de chacune de ses phrases.

    Args:
        phrases : découpage du texte (voir cache.decouper_pour_cache)
        tokens  : sortie du calculateur pour ce texte : tableaux "debut" (position du
                  token dans le texte), "logprob", "top1", "entropie"
    Returns:
        (features du paragraphe (N_FEATURES_PPL,),
         features par phrase (len(phrases), N_FEATURES_PPL_PHRASE))
    """
    lp = np.asarray(tokens["logprob"], dtype=np.float64)
    t1 = np.asarray(tokens["top1"], dtype=np.float64)
    ent = np.asarray(tokens["entropie"], dtype=np.float64)
    global_ = _stats(lp, t1, ent)

    # Chaque token appartient à la dernière phrase commencée avant lui
    num = np.searchsorted(bornes_phrases(texte, phrases), np.asarray(tokens["debut"]),
                          side="right") - 1
    num = np.clip(num, 0, len(phrases) - 1)
    par_phrase = np.empty((len(phrases), N_FEATURES_PPL_PHRASE), dtype=np.float32)
    avec_tokens = []
    for j in range(len(phrases)):
        m = num == j
        if m.any():
            par_phrase[j] = _stats(lp[m], t1[m], ent[m])
            avec_tokens.append(j)
        else:                         # phrase tronquée (au-delà de max_tokens) : valeurs du paragraphe
            par_phrase[j] = global_

    moy_phrases = par_phrase[avec_tokens, 0]
    paragraphe = np.array([
        *global_,
        -global_[0] / max(global_[3], 1e-6) if len(lp) else 0.0,
        moy_phrases.std() if len(moy_phrases) > 1 else 0.0,
    ], dtype=np.float32)
    return paragraphe, par_phrase


def calculer_tokens(textes, nom_modele: str, max_length: int = 256,
                    tokens_par_lot: int = 8192, appareil: str | None = None) -> list[dict]:
    """
    Passe chaque texte dans le modèle de langue causal et retourne, pour chaque
    texte, les statistiques de chacun de ses tokens (voir features_perplexite).

    Les textes sont triés par longueur et regroupés en lots d'environ
    `tokens_par_lot` tokens (moins de padding, mémoire GPU maîtrisée).
    """
    import torch
    import torch.nn.functional as F

    from ..common.models import get_modele_causal

    textes = list(textes)
    if not textes:
        return []
    tokenizer, modele, dev = get_modele_causal(nom_modele, appareil)
    debut_seq = tokenizer.bos_token_id if tokenizer.bos_token_id is not None \
        else tokenizer.eos_token_id
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else debut_seq

    enc = tokenizer(textes, add_special_tokens=False, truncation=True,
                    max_length=max_length - 1, return_offsets_mapping=True)
    ids = [[debut_seq] + x for x in enc["input_ids"]]       # le 1er token est prédit lui aussi
    ordre = np.argsort([len(x) for x in ids])
    sortie = [None] * len(textes)

    i = 0
    with torch.no_grad():
        while i < len(ordre):
            lot = ordre[i:i + max(1, tokens_par_lot // len(ids[ordre[i]]))]
            longueur = len(ids[lot[-1]])                    # textes triés : le dernier est le plus long
            lot = lot[:max(1, tokens_par_lot // longueur)]
            longueur = len(ids[lot[-1]])
            x = torch.full((len(lot), longueur), pad, dtype=torch.long)
            masque = torch.zeros((len(lot), longueur), dtype=torch.long)
            for r, k in enumerate(lot):
                x[r, :len(ids[k])] = torch.tensor(ids[k])
                masque[r, :len(ids[k])] = 1
            x, masque = x.to(dev), masque.to(dev)
            ctx = (torch.autocast(device_type="cuda", dtype=torch.float16)
                   if dev.type == "cuda" else contextlib.nullcontext())
            with ctx:
                logits = modele(input_ids=x, attention_mask=masque).logits
            for r, k in enumerate(lot):
                n = len(ids[k]) - 1                         # tokens réels à prédire
                lp = F.log_softmax(logits[r, :n].float(), dim=-1)
                cible = x[r, 1:n + 1]
                sortie[k] = {
                    "debut": np.array([o[0] for o in enc["offset_mapping"][k]], dtype=np.int32),
                    "logprob": lp.gather(1, cible[:, None])[:, 0].cpu().numpy(),
                    "top1": (lp.argmax(dim=-1) == cible).cpu().numpy(),
                    "entropie": -(lp.exp() * lp).sum(dim=-1).cpu().numpy(),
                }
            del logits
            i += len(lot)
    return sortie
