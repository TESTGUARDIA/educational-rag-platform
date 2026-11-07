from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from jose import JWTError, jwt
from passlib.context import CryptContext
import sqlite3, json, datetime, logging, os, shutil, uuid
from contextlib import asynccontextmanager

from src.web.back import services

log = logging.getLogger("testguardai.api")

# ─── CONFIG ────────────────────────────────────────────────────────────────────
SECRET = os.environ.get("JWT_SECRET")
if not SECRET:
    raise RuntimeError("JWT_SECRET manquant — définissez-le dans .env")

ALGO         = "HS256"
DB           = os.environ.get("DB_PATH", "testguard.db")
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
UPLOADS_DIR  = os.environ.get("UPLOADS_DIR", "./uploads")
MAX_PDF_SIZE = 20 * 1024 * 1024  # 20 Mo (ING-01)
os.makedirs(UPLOADS_DIR, exist_ok=True)

NB_QUESTIONS_DEFAUT = int(os.environ.get("NUM_QUESTIONS", "5"))
NB_QUESTIONS_MAX    = 30
SEUIL_PAR_DEFAUT    = 0.5   # rapports enregistrés avant J1, sans seuil de modèle

@asynccontextmanager
async def cycle_de_vie(_app):
    """Au démarrage : crée/migre la base, puis précharge les modèles (si demandé)."""
    init_db()
    services.demarrer_prechargement()
    yield

app = FastAPI(title="TestGuardAI API", lifespan=cycle_de_vie)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)
pwd    = CryptContext(schemes=["bcrypt"])
oauth2 = OAuth2PasswordBearer(tokenUrl="/auth/login")

# ─── DB ────────────────────────────────────────────────────────────────────────
def get_db():
    # check_same_thread=False : FastAPI ouvre la connexion dans un thread du pool
    # mais l'utilise dans la boucle événementielle pour les routes async
    # (ex. POST /documents). Une connexion par requête, jamais partagée.
    con = sqlite3.connect(DB, check_same_thread=False)
    con.row_factory = sqlite3.Row
    try: yield con
    finally: con.close()

def init_db():
    con = sqlite3.connect(DB)
    c = con.cursor()
    c.executescript("""
    CREATE TABLE IF NOT EXISTS Utilisateur (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        nom TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        mot_de_passe_hash TEXT NOT NULL,
        role TEXT NOT NULL CHECK(role IN ('teacher','student')),
        date_creation TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    CREATE TABLE IF NOT EXISTS Devoir (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        titre TEXT NOT NULL,
        description TEXT,
        document_pdf_source TEXT,
        date_limite TIMESTAMP,
        cree_par_id INTEGER REFERENCES Utilisateur(id)
    );
    CREATE TABLE IF NOT EXISTS Question (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        devoir_id INTEGER REFERENCES Devoir(id),
        numero_ordre INTEGER,
        enonce TEXT NOT NULL,
        reponse_attendue TEXT,
        points_attribues REAL DEFAULT 1.0
    );
    CREATE TABLE IF NOT EXISTS Soumission (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        devoir_id INTEGER REFERENCES Devoir(id),
        etudiant_id INTEGER REFERENCES Utilisateur(id),
        date_depot TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        statut_global TEXT DEFAULT 'soumis',
        note_finale_humaine REAL
    );
    CREATE TABLE IF NOT EXISTS ReponseEtudiant (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        soumission_id INTEGER REFERENCES Soumission(id),
        question_id INTEGER REFERENCES Question(id),
        texte_etudiant TEXT
    );
    CREATE TABLE IF NOT EXISTS CorrectionAutomatique (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        soumission_id INTEGER REFERENCES Soumission(id),
        note_ia_suggeree REAL,
        statut_correction TEXT DEFAULT 'en_attente',
        commentaires_ia TEXT,
        details TEXT
    );
    CREATE TABLE IF NOT EXISTS RapportDetectionIA (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        soumission_id INTEGER REFERENCES Soumission(id),
        score_ia_global REAL,
        statut_analyse TEXT DEFAULT 'non_analysé',
        details_phrases_suspectes TEXT,
        seuil REAL,
        langue TEXT,
        message TEXT
    );
    CREATE TABLE IF NOT EXISTS Document (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        nom_original TEXT NOT NULL,
        chemin_stockage TEXT NOT NULL,
        taille_octets INTEGER NOT NULL,
        uploade_par_id INTEGER REFERENCES Utilisateur(id),
        date_upload TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
    # Migrations : colonnes ajoutées à J1 sur des bases créées avant
    for table, colonne, type_ in (("RapportDetectionIA", "seuil", "REAL"),
                                  ("RapportDetectionIA", "langue", "TEXT"),
                                  ("RapportDetectionIA", "message", "TEXT"),
                                  ("CorrectionAutomatique", "details", "TEXT")):
        colonnes = [r[1] for r in c.execute(f"PRAGMA table_info({table})")]
        if colonne not in colonnes:
            c.execute(f"ALTER TABLE {table} ADD COLUMN {colonne} {type_}")
    # Seed utilisateurs
    c.execute("SELECT COUNT(*) FROM Utilisateur")
    if c.fetchone()[0] == 0:
        users = [
            ("Marie Dupont",   "marie.dupont@testguard.ai",   pwd.hash("password"), "teacher"),
            ("Lucas Moreau",   "lucas.moreau@testguard.ai",   pwd.hash("password"), "student"),
            ("Emma Lefebvre",  "emma.lefebvre@testguard.ai",  pwd.hash("password"), "student"),
            ("Hugo Bernard",   "hugo.bernard@testguard.ai",   pwd.hash("password"), "student"),
            ("Chloé Martin",   "chloe.martin@testguard.ai",   pwd.hash("password"), "student"),
        ]
        c.executemany("INSERT INTO Utilisateur(nom,email,mot_de_passe_hash,role) VALUES(?,?,?,?)", users)
        # Seed devoir
        c.execute("INSERT INTO Devoir(titre,description,date_limite,cree_par_id) VALUES(?,?,?,1)",
                  ("Mathématiques – Intégrales", "Calcul intégral et sommes de Riemann",
                   "2027-06-30 23:59:00"))
        devoir_id = c.lastrowid
        questions = [
            ("Définissez l'intégrale de Riemann.", "Limite des sommes de Riemann quand n→∞", 4.0),
            ("Calculez ∫₀¹ x² dx.", "1/3", 6.0),
            ("Énoncer le théorème fondamental de l'analyse.", "F'(x)=f(x) si F=∫f", 5.0),
            ("Donnez un exemple d'application des intégrales.", "Aire sous une courbe, etc.", 5.0),
        ]
        for i, (e, r, p) in enumerate(questions, 1):
            c.execute("INSERT INTO Question(devoir_id,numero_ordre,enonce,reponse_attendue,points_attribues) VALUES(?,?,?,?,?)",
                      (devoir_id, i, e, r, p))
        students = [(2, None, 17, 2), (3, None, None, 84), (4, None, None, 12), (5, None, 15, 5)]
        for uid, _, note, ia_score in students:
            c.execute("INSERT INTO Soumission(devoir_id,etudiant_id,statut_global,note_finale_humaine) VALUES(?,?,?,?)",
                      (devoir_id, uid, 'noté' if note else 'soumis', note))
            s_id = c.lastrowid
            c.execute("INSERT INTO RapportDetectionIA(soumission_id,score_ia_global,statut_analyse) VALUES(?,?,?)",
                      (s_id, ia_score/100, 'analysé'))
            if note:
                c.execute("INSERT INTO CorrectionAutomatique(soumission_id,note_ia_suggeree,statut_correction,commentaires_ia) VALUES(?,?,?,?)",
                          (s_id, note-0.5, 'validé', 'Réponses cohérentes et bien structurées.'))
    con.commit()
    con.close()

# ─── AUTH ───────────────────────────────────────────────────────────────────────
def make_tokens(uid: int, role: str):
    """Génère un access token (15 min) et un refresh token (7 jours)."""
    now = datetime.datetime.utcnow()
    access = jwt.encode(
        {"sub": str(uid), "role": role, "type": "access",
         "exp": now + datetime.timedelta(minutes=15)},
        SECRET, ALGO
    )
    refresh = jwt.encode(
        {"sub": str(uid), "role": role, "type": "refresh",
         "exp": now + datetime.timedelta(days=7)},
        SECRET, ALGO
    )
    return access, refresh

def current_user(token: str = Depends(oauth2), db=Depends(get_db)):
    try:
        payload = jwt.decode(token, SECRET, algorithms=[ALGO])
        if payload.get("type") != "access":
            raise HTTPException(401, "Type de token invalide")
        uid = int(payload["sub"])
    except JWTError:
        raise HTTPException(status_code=401, detail="Token invalide ou expiré")
    row = db.execute("SELECT * FROM Utilisateur WHERE id=?", (uid,)).fetchone()
    if not row:
        raise HTTPException(404, "Utilisateur introuvable")
    return dict(row)

# ─── SCHEMAS ────────────────────────────────────────────────────────────────────
class GradeIn(BaseModel):
    note: float

class DevoirIn(BaseModel):
    titre: str
    description: str = ""
    date_limite: str = ""

class ReponseIn(BaseModel):
    question_id: int
    texte: str

class SoumissionIn(BaseModel):
    devoir_id: int
    reponses: list[ReponseIn]

class RefreshIn(BaseModel):
    refresh_token: str

# ─── ROUTES AUTH ────────────────────────────────────────────────────────────────
@app.post("/auth/login")
def login(form: OAuth2PasswordRequestForm = Depends(), db=Depends(get_db)):
    row = db.execute("SELECT * FROM Utilisateur WHERE email=?", (form.username,)).fetchone()
    if not row or not pwd.verify(form.password, row["mot_de_passe_hash"]):
        raise HTTPException(401, "Identifiants incorrects")
    access, refresh = make_tokens(row["id"], row["role"])
    return {
        "access_token":  access,
        "refresh_token": refresh,
        "token_type":    "bearer",
        "role":          row["role"],
        "nom":           row["nom"],
        "id":            row["id"],
    }

@app.post("/auth/refresh")
def refresh_token(body: RefreshIn):
    """Émet un nouvel access token à partir d'un refresh token valide."""
    try:
        payload = jwt.decode(body.refresh_token, SECRET, algorithms=[ALGO])
        if payload.get("type") != "refresh":
            raise HTTPException(401, "Token de refresh invalide")
        uid  = int(payload["sub"])
        role = payload["role"]
    except JWTError:
        raise HTTPException(401, "Refresh token expiré ou invalide — reconnexion requise")
    access, new_refresh = make_tokens(uid, role)
    return {"access_token": access, "refresh_token": new_refresh, "token_type": "bearer"}

@app.get("/me")
def me(u=Depends(current_user)):
    return u

# ─── DEVOIRS ────────────────────────────────────────────────────────────────────
@app.get("/devoirs")
def list_devoirs(u=Depends(current_user), db=Depends(get_db)):
    if u["role"] == "teacher":
        rows = db.execute("SELECT * FROM Devoir WHERE cree_par_id=?", (u["id"],)).fetchall()
    else:
        rows = db.execute("SELECT * FROM Devoir").fetchall()
    result = []
    for r in rows:
        d = dict(r)
        d["questions"] = [dict(q) for q in db.execute(
            "SELECT * FROM Question WHERE devoir_id=?", (r["id"],)).fetchall()]
        result.append(d)
    return result

@app.post("/devoirs")
def create_devoir(body: DevoirIn, u=Depends(current_user), db=Depends(get_db)):
    if u["role"] != "teacher":
        raise HTTPException(403, "Réservé aux enseignants")
    c = db.cursor()
    c.execute("INSERT INTO Devoir(titre,description,date_limite,cree_par_id) VALUES(?,?,?,?)",
              (body.titre, body.description, body.date_limite, u["id"]))
    db.commit()
    return {"id": c.lastrowid, "message": "Devoir créé"}

@app.get("/devoirs/{did}/questions")
def get_questions(did: int, db=Depends(get_db), u=Depends(current_user)):
    return [dict(r) for r in db.execute(
        "SELECT * FROM Question WHERE devoir_id=? ORDER BY numero_ordre", (did,)).fetchall()]

def _enregistrer_pdf(db, nom: str, contenu: bytes, uid: int) -> int:
    """Écrit le PDF dans UPLOADS_DIR et l'enregistre dans Document. Retourne son id."""
    chemin = os.path.join(UPLOADS_DIR, f"{uuid.uuid4().hex}.pdf")
    with open(chemin, "wb") as out:
        out.write(contenu)
    c = db.execute(
        "INSERT INTO Document(nom_original, chemin_stockage, taille_octets, uploade_par_id) VALUES(?,?,?,?)",
        (nom, chemin, len(contenu), uid))
    return c.lastrowid

@app.post("/devoirs/{did}/generer")
def generer_devoir(
    did: int,
    fichier: UploadFile | None = File(None),
    document_id: int | None = Form(None),
    nb_questions: int = Form(NB_QUESTIONS_DEFAUT),
    u=Depends(current_user),
    db=Depends(get_db),
):
    """
    Génère les questions d'un devoir à partir d'un cours PDF (pôle Génération).
    Le PDF est soit envoyé (`fichier`), soit déjà déposé (`document_id`).
    Les questions sont ajoutées à la suite de celles du devoir.
    Route synchrone : la génération (Ollama) prend de quelques secondes à quelques minutes.
    """
    if u["role"] != "teacher":
        raise HTTPException(403, "Réservé aux enseignants")
    devoir = db.execute("SELECT * FROM Devoir WHERE id=?", (did,)).fetchone()
    if not devoir or devoir["cree_par_id"] != u["id"]:
        raise HTTPException(404, "Devoir introuvable ou accès refusé")
    if not 1 <= nb_questions <= NB_QUESTIONS_MAX:
        raise HTTPException(400, f"nb_questions doit être entre 1 et {NB_QUESTIONS_MAX}")

    if fichier is not None:
        contenu = fichier.file.read(MAX_PDF_SIZE + 1)
        if contenu[:4] != b"%PDF":
            raise HTTPException(400, "Format invalide — ce fichier n'est pas un PDF")
        if len(contenu) > MAX_PDF_SIZE:
            raise HTTPException(400, "Fichier trop volumineux — maximum 20 Mo")
        nom = fichier.filename or "cours.pdf"
        document_id = _enregistrer_pdf(db, nom, contenu, u["id"])
    elif document_id is not None:
        doc = db.execute("SELECT * FROM Document WHERE id=? AND uploade_par_id=?",
                         (document_id, u["id"])).fetchone()
        if not doc or not os.path.exists(doc["chemin_stockage"]):
            raise HTTPException(404, "Document introuvable ou accès refusé")
        with open(doc["chemin_stockage"], "rb") as f:
            contenu = f.read()
        nom = doc["nom_original"]
    else:
        raise HTTPException(400, "Envoyez un PDF (fichier) ou l'id d'un document déposé (document_id)")

    try:
        generees = services.generer_questions(contenu, nom, nb_questions)
    except Exception as e:
        log.exception("Génération impossible pour le devoir %s", did)
        raise HTTPException(502, f"Génération impossible : {type(e).__name__} : {e}")
    if not generees:
        raise HTTPException(502, "Le modèle n'a produit aucune question exploitable — réessayez.")

    dernier = db.execute("SELECT COALESCE(MAX(numero_ordre), 0) FROM Question WHERE devoir_id=?",
                         (did,)).fetchone()[0]
    for i, q in enumerate(generees, dernier + 1):
        db.execute("INSERT INTO Question(devoir_id,numero_ordre,enonce,reponse_attendue,points_attribues) VALUES(?,?,?,?,1.0)",
                   (did, i, q["question"], q["reponse"]))
    db.execute("UPDATE Devoir SET document_pdf_source=? WHERE id=?", (str(document_id), did))
    db.commit()
    questions = [dict(r) for r in db.execute(
        "SELECT * FROM Question WHERE devoir_id=? ORDER BY numero_ordre", (did,)).fetchall()]
    return {"devoir_id": did, "document_id": document_id,
            "nb_generees": len(generees), "questions": questions}

# ─── SOUMISSIONS ────────────────────────────────────────────────────────────────
@app.get("/soumissions")
def list_soumissions(devoir_id: int = None, u=Depends(current_user), db=Depends(get_db)):
    if u["role"] == "teacher":
        q      = "SELECT s.*, u.nom as etudiant_nom FROM Soumission s JOIN Utilisateur u ON s.etudiant_id=u.id"
        params = ()
        if devoir_id:
            q      += " WHERE s.devoir_id=?"
            params  = (devoir_id,)
    else:
        q      = "SELECT s.*, u.nom as etudiant_nom FROM Soumission s JOIN Utilisateur u ON s.etudiant_id=u.id WHERE s.etudiant_id=?"
        params = (u["id"],)
    rows = [dict(r) for r in db.execute(q, params).fetchall()]
    for r in rows:
        ia   = db.execute("SELECT * FROM RapportDetectionIA WHERE soumission_id=?",    (r["id"],)).fetchone()
        corr = db.execute("SELECT * FROM CorrectionAutomatique WHERE soumission_id=?", (r["id"],)).fetchone()
        r["ia"]         = dict(ia)   if ia   else None
        r["correction"] = dict(corr) if corr else None
    return rows

def _textes_soumission(db, sid: int) -> list[str]:
    return [r["texte_etudiant"] or "" for r in db.execute(
        "SELECT texte_etudiant FROM ReponseEtudiant WHERE soumission_id=? ORDER BY id", (sid,))]

def _enregistrer_detection(db, sid: int, res: dict) -> None:
    """Crée ou met à jour le rapport de détection d'une soumission."""
    valeurs = (res.get("score"), res["statut"],
               json.dumps(res.get("phrases", []), ensure_ascii=False),
               res.get("seuil"), res.get("langue"),
               res.get("raison") or " ".join(res.get("avertissements", [])) or None)
    if db.execute("SELECT id FROM RapportDetectionIA WHERE soumission_id=?", (sid,)).fetchone():
        db.execute("""UPDATE RapportDetectionIA SET score_ia_global=?, statut_analyse=?,
                      details_phrases_suspectes=?, seuil=?, langue=?, message=?
                      WHERE soumission_id=?""", (*valeurs, sid))
    else:
        db.execute("""INSERT INTO RapportDetectionIA(score_ia_global, statut_analyse,
                      details_phrases_suspectes, seuil, langue, message, soumission_id)
                      VALUES(?,?,?,?,?,?,?)""", (*valeurs, sid))
    db.commit()

def _analyser_en_arriere_plan(sid: int) -> None:
    """Détection IA lancée après le dépôt d'une copie (connexion SQLite propre au thread)."""
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    try:
        _enregistrer_detection(con, sid, services.detecter_ia(_textes_soumission(con, sid)))
    except Exception:
        log.exception("Détection IA en arrière-plan impossible (soumission %s)", sid)
    finally:
        con.close()

@app.post("/soumissions")
def submit(body: SoumissionIn, taches: BackgroundTasks,
           u=Depends(current_user), db=Depends(get_db)):
    if u["role"] != "student":
        raise HTTPException(403, "Réservé aux étudiants")
    if not db.execute("SELECT id FROM Devoir WHERE id=?", (body.devoir_id,)).fetchone():
        raise HTTPException(404, "Devoir introuvable")
    c = db.cursor()
    c.execute("INSERT INTO Soumission(devoir_id,etudiant_id,statut_global) VALUES(?,?,?)",
              (body.devoir_id, u["id"], "soumis"))
    s_id = c.lastrowid
    for r in body.reponses:
        c.execute("INSERT INTO ReponseEtudiant(soumission_id,question_id,texte_etudiant) VALUES(?,?,?)",
                  (s_id, r.question_id, r.texte))
    # Détection IA (pôle Évaluation) après la réponse : l'étudiant n'attend pas le modèle
    c.execute("INSERT INTO RapportDetectionIA(soumission_id,statut_analyse) VALUES(?,?)",
              (s_id, "en_cours"))
    db.commit()
    taches.add_task(_analyser_en_arriere_plan, s_id)
    return {"id": s_id, "statut_analyse": "en_cours"}

@app.post("/soumissions/{sid}/corriger-auto")
def corriger_auto(sid: int, u=Depends(current_user), db=Depends(get_db)):
    """
    Note suggérée par le pôle Évaluation : chaque réponse reçoit un score [0, 1]
    pondéré par les points de la question, ramené sur 20. L'enseignant valide
    ensuite avec /soumissions/{sid}/noter.
    """
    if u["role"] != "teacher":
        raise HTTPException(403, "Réservé aux enseignants")
    if not db.execute("SELECT id FROM Soumission WHERE id=?", (sid,)).fetchone():
        raise HTTPException(404, "Soumission introuvable")
    lignes = db.execute("""
        SELECT re.question_id, re.texte_etudiant, q.enonce, q.reponse_attendue,
               q.points_attribues, q.numero_ordre
        FROM ReponseEtudiant re JOIN Question q ON re.question_id = q.id
        WHERE re.soumission_id = ? ORDER BY q.numero_ordre""", (sid,)).fetchall()

    details, obtenus, total = [], 0.0, 0.0
    try:
        for l in lignes:
            if not (l["reponse_attendue"] or "").strip():
                details.append({"question_id": l["question_id"], "numero": l["numero_ordre"],
                                "statut": "non_corrigée", "raison": "Pas de réponse attendue"})
                continue
            pts = float(l["points_attribues"] or 1.0)
            r = services.corriger_reponse(l["enonce"], l["texte_etudiant"] or "", l["reponse_attendue"])
            obtenus += r["score"] * pts
            total += pts
            details.append({"question_id": l["question_id"], "numero": l["numero_ordre"],
                            "score": round(r["score"], 4), "correct": r["correct"],
                            "verdict": r["verdict"], "points": pts,
                            "points_obtenus": round(r["score"] * pts, 2)})
    except Exception as e:
        log.exception("Correction automatique impossible (soumission %s)", sid)
        raise HTTPException(502, f"Correction impossible : {type(e).__name__} : {e}")
    if total == 0:
        raise HTTPException(400, "Aucune question corrigeable (réponses attendues manquantes)")

    note = round(obtenus / total * 20 * 2) / 2          # au demi-point, comme la saisie
    n_ok = sum(1 for d in details if d.get("correct"))
    n_corr = sum(1 for d in details if "score" in d)
    commentaire = f"{n_ok}/{n_corr} réponse(s) jugée(s) correcte(s) par la correction automatique."
    valeurs = (note, "en_attente", commentaire, json.dumps(details, ensure_ascii=False))
    if db.execute("SELECT id FROM CorrectionAutomatique WHERE soumission_id=?", (sid,)).fetchone():
        db.execute("""UPDATE CorrectionAutomatique SET note_ia_suggeree=?, statut_correction=?,
                      commentaires_ia=?, details=? WHERE soumission_id=?""", (*valeurs, sid))
    else:
        db.execute("""INSERT INTO CorrectionAutomatique(note_ia_suggeree, statut_correction,
                      commentaires_ia, details, soumission_id) VALUES(?,?,?,?,?)""", (*valeurs, sid))
    db.commit()
    return {"soumission_id": sid, "note_ia_suggeree": note, "commentaires_ia": commentaire,
            "details": details}

@app.post("/soumissions/{sid}/noter")
def noter(sid: int, body: GradeIn, u=Depends(current_user), db=Depends(get_db)):
    if u["role"] != "teacher":
        raise HTTPException(403, "Réservé aux enseignants")
    db.execute("UPDATE Soumission SET note_finale_humaine=?, statut_global='noté' WHERE id=?",
               (body.note, sid))
    db.execute("UPDATE CorrectionAutomatique SET statut_correction='validé' WHERE soumission_id=?", (sid,))
    db.commit()
    return {"ok": True}

@app.post("/soumissions/{sid}/analyser-ia")
def analyser_ia(sid: int, u=Depends(current_user), db=Depends(get_db)):
    if u["role"] != "teacher":
        raise HTTPException(403, "Réservé aux enseignants")
    if not db.execute("SELECT id FROM Soumission WHERE id=?", (sid,)).fetchone():
        raise HTTPException(404, "Soumission introuvable")
    res = services.detecter_ia(_textes_soumission(db, sid))
    _enregistrer_detection(db, sid, res)
    return {"statut_analyse": res["statut"], "score_ia_global": res.get("score"),
            "seuil": res.get("seuil"), "langue": res.get("langue"),
            "phrases": res.get("phrases", []), "message": res.get("raison")
            or " ".join(res.get("avertissements", [])) or None}

@app.get("/soumissions/{sid}/reponses")
def get_reponses(sid: int, u=Depends(current_user), db=Depends(get_db)):
    rows = db.execute("""
        SELECT re.*, q.enonce, q.numero_ordre, q.points_attribues
        FROM ReponseEtudiant re
        JOIN Question q ON re.question_id = q.id
        WHERE re.soumission_id = ?
        ORDER BY q.numero_ordre
    """, (sid,)).fetchall()
    return [dict(r) for r in rows]

# ─── DOCUMENTS (upload PDF) ─────────────────────────────────────────────────────
@app.post("/documents")
async def upload_documents(
    files: list[UploadFile] = File(...),
    u=Depends(current_user),
    db=Depends(get_db)
):
    """
    Upload jusqu'à 10 fichiers PDF simultanément (≤ 20 Mo chacun).
    Validation magic bytes : vérifie que chaque fichier est bien un PDF,
    indépendamment de son extension (ING-01 / ENF-03).
    """
    if u["role"] != "teacher":
        raise HTTPException(403, "Réservé aux enseignants")
    if len(files) > 10:
        raise HTTPException(400, "Maximum 10 fichiers par envoi")

    saved = []
    errors = []

    for f in files:
        # Lire les 4 premiers octets pour valider le magic bytes PDF (%PDF)
        header = await f.read(4)
        if header != b"%PDF":
            errors.append({
                "fichier": f.filename,
                "erreur": "Format invalide — ce fichier n'est pas un PDF"
            })
            continue

        # Vérifier la taille (remettre le curseur au début pour lire le reste)
        await f.seek(0)
        contenu = await f.read()
        if len(contenu) > MAX_PDF_SIZE:
            errors.append({
                "fichier": f.filename,
                "erreur": f"Fichier trop volumineux ({len(contenu)//1024//1024} Mo) — maximum 20 Mo"
            })
            continue

        # Sauvegarder avec un nom unique pour éviter les collisions
        ext       = os.path.splitext(f.filename)[1] or ".pdf"
        nom_safe  = f"{uuid.uuid4().hex}{ext}"
        chemin    = os.path.join(UPLOADS_DIR, nom_safe)
        with open(chemin, "wb") as out:
            out.write(contenu)

        # Enregistrer en base
        c = db.cursor()
        c.execute(
            "INSERT INTO Document(nom_original, chemin_stockage, taille_octets, uploade_par_id) VALUES(?,?,?,?)",
            (f.filename, chemin, len(contenu), u["id"])
        )
        db.commit()
        saved.append({"id": c.lastrowid, "nom": f.filename, "taille": len(contenu)})

    return {
        "sauvegardes": saved,
        "erreurs":     errors,
        "message":     f"{len(saved)} fichier(s) uploadé(s), {len(errors)} erreur(s)"
    }

@app.get("/documents")
def list_documents(u=Depends(current_user), db=Depends(get_db)):
    """Liste les documents uploadés par l'enseignant connecté."""
    if u["role"] != "teacher":
        raise HTTPException(403, "Réservé aux enseignants")
    rows = db.execute(
        "SELECT id, nom_original, taille_octets, date_upload FROM Document WHERE uploade_par_id=? ORDER BY date_upload DESC",
        (u["id"],)
    ).fetchall()
    return [dict(r) for r in rows]

@app.delete("/documents/{doc_id}")
def delete_document(doc_id: int, u=Depends(current_user), db=Depends(get_db)):
    """Supprime un document (fichier + entrée BDD)."""
    if u["role"] != "teacher":
        raise HTTPException(403, "Réservé aux enseignants")
    row = db.execute(
        "SELECT * FROM Document WHERE id=? AND uploade_par_id=?", (doc_id, u["id"])
    ).fetchone()
    if not row:
        raise HTTPException(404, "Document introuvable ou accès refusé")
    # Supprimer le fichier physique
    if os.path.exists(row["chemin_stockage"]):
        os.remove(row["chemin_stockage"])
    db.execute("DELETE FROM Document WHERE id=?", (doc_id,))
    db.commit()
    return {"ok": True, "message": f"Document {row['nom_original']} supprimé"}

# ─── STATS ──────────────────────────────────────────────────────────────────────
@app.get("/stats")
def stats(u=Depends(current_user), db=Depends(get_db)):
    if u["role"] != "teacher":
        raise HTTPException(403, "Réservé aux enseignants")
    total    = db.execute("SELECT COUNT(*) FROM Soumission").fetchone()[0]
    # Seuil appris par le modèle de détection (faux positifs bornés), pas 0.5
    detected = db.execute(
        "SELECT COUNT(*) FROM RapportDetectionIA WHERE statut_analyse='analysé' "
        "AND score_ia_global >= COALESCE(seuil, ?)", (SEUIL_PAR_DEFAUT,)).fetchone()[0]
    avg_row  = db.execute("SELECT AVG(note_finale_humaine) FROM Soumission WHERE note_finale_humaine IS NOT NULL").fetchone()[0]
    pending  = db.execute("SELECT COUNT(*) FROM Soumission WHERE statut_global='soumis'").fetchone()[0]
    return {"total": total, "detected": detected, "moyenne": round(avg_row or 0, 1), "pending": pending}

# ─── ENTRYPOINT ─────────────────────────────────────────────────────────────────
# Depuis la racine du dépôt :  uvicorn src.web.back.main:app --port 8000
#                         ou   python -m src.web.back.main
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
