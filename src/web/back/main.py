from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from jose import JWTError, jwt
from passlib.context import CryptContext
import sqlite3, json, random, datetime, os

SECRET = "testguardai-secret-key-2024"
ALGO = "HS256"
DB = "testguard.db"

app = FastAPI(title="TestGuardAI API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
pwd = CryptContext(schemes=["bcrypt"])
oauth2 = OAuth2PasswordBearer(tokenUrl="/auth/login")

# ─── DB ───────────────────────────────────────────────────────────────────────
def get_db():
    con = sqlite3.connect(DB)
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
        commentaires_ia TEXT
    );
    CREATE TABLE IF NOT EXISTS RapportDetectionIA (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        soumission_id INTEGER REFERENCES Soumission(id),
        score_ia_global REAL,
        statut_analyse TEXT DEFAULT 'non_analyse',
        details_phrases_suspectes TEXT
    );
    """)
    # Seed users
    c.execute("SELECT COUNT(*) FROM Utilisateur")
    if c.fetchone()[0] == 0:
        users = [
            ("Marie Dupont", "marie.dupont@testguard.ai", pwd.hash("password"), "teacher"),
            ("Lucas Moreau", "lucas.moreau@testguard.ai", pwd.hash("password"), "student"),
            ("Emma Lefebvre", "emma.lefebvre@testguard.ai", pwd.hash("password"), "student"),
            ("Hugo Bernard", "hugo.bernard@testguard.ai", pwd.hash("password"), "student"),
            ("Chloé Martin", "chloe.martin@testguard.ai", pwd.hash("password"), "student"),
        ]
        c.executemany("INSERT INTO Utilisateur(nom,email,mot_de_passe_hash,role) VALUES(?,?,?,?)", users)
        # Seed devoir
        c.execute("INSERT INTO Devoir(titre,description,date_limite,cree_par_id) VALUES(?,?,?,1)",
                  ("Mathématiques – Intégrales", "Calcul intégral et sommes de Riemann",
                   "2025-06-30 23:59:00"))
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
        # Seed soumissions
        students = [(2,"Lucas Moreau",17,2),(3,"Emma Lefebvre",None,84),(4,"Hugo Bernard",None,12),(5,"Chloé Martin",15,5)]
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

# ─── AUTH ──────────────────────────────────────────────────────────────────────
def make_token(uid: int, role: str):
    exp = datetime.datetime.utcnow() + datetime.timedelta(hours=24)
    return jwt.encode({"sub": str(uid), "role": role, "exp": exp}, SECRET, ALGO)

def current_user(token: str = Depends(oauth2), db=Depends(get_db)):
    try:
        payload = jwt.decode(token, SECRET, algorithms=[ALGO])
        uid = int(payload["sub"])
    except JWTError:
        raise HTTPException(status_code=401, detail="Token invalide")
    row = db.execute("SELECT * FROM Utilisateur WHERE id=?", (uid,)).fetchone()
    if not row: raise HTTPException(404, "Utilisateur introuvable")
    return dict(row)

# ─── SCHEMAS ───────────────────────────────────────────────────────────────────
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

# ─── ROUTES ────────────────────────────────────────────────────────────────────
@app.post("/auth/login")
def login(form: OAuth2PasswordRequestForm = Depends(), db=Depends(get_db)):
    row = db.execute("SELECT * FROM Utilisateur WHERE email=?", (form.username,)).fetchone()
    if not row or not pwd.verify(form.password, row["mot_de_passe_hash"]):
        raise HTTPException(401, "Identifiants incorrects")
    return {"access_token": make_token(row["id"], row["role"]), "role": row["role"],
            "nom": row["nom"], "id": row["id"]}

@app.get("/me")
def me(u=Depends(current_user)): return u

# ─── DEVOIRS ───────────────────────────────────────────────────────────────────
@app.get("/devoirs")
def list_devoirs(u=Depends(current_user), db=Depends(get_db)):
    if u["role"] == "teacher":
        rows = db.execute("SELECT * FROM Devoir WHERE cree_par_id=?", (u["id"],)).fetchall()
    else:
        rows = db.execute("SELECT * FROM Devoir").fetchall()
    result = []
    for r in rows:
        d = dict(r)
        d["questions"] = [dict(q) for q in db.execute("SELECT * FROM Question WHERE devoir_id=?", (r["id"],)).fetchall()]
        result.append(d)
    return result

@app.post("/devoirs")
def create_devoir(body: DevoirIn, u=Depends(current_user), db=Depends(get_db)):
    if u["role"] != "teacher": raise HTTPException(403)
    c = db.cursor()
    c.execute("INSERT INTO Devoir(titre,description,date_limite,cree_par_id) VALUES(?,?,?,?)",
              (body.titre, body.description, body.date_limite, u["id"]))
    db.commit()
    return {"id": c.lastrowid, "message": "Devoir créé"}

@app.get("/devoirs/{did}/questions")
def get_questions(did: int, db=Depends(get_db), u=Depends(current_user)):
    return [dict(r) for r in db.execute("SELECT * FROM Question WHERE devoir_id=? ORDER BY numero_ordre", (did,)).fetchall()]

# ─── SOUMISSIONS ───────────────────────────────────────────────────────────────
@app.get("/soumissions")
def list_soumissions(devoir_id: int = None, u=Depends(current_user), db=Depends(get_db)):
    if u["role"] == "teacher":
        q = "SELECT s.*, u.nom as etudiant_nom FROM Soumission s JOIN Utilisateur u ON s.etudiant_id=u.id"
        params = ()
        if devoir_id:
            q += " WHERE s.devoir_id=?"
            params = (devoir_id,)
    else:
        q = "SELECT s.*, u.nom as etudiant_nom FROM Soumission s JOIN Utilisateur u ON s.etudiant_id=u.id WHERE s.etudiant_id=?"
        params = (u["id"],)
    rows = [dict(r) for r in db.execute(q, params).fetchall()]
    for r in rows:
        ia = db.execute("SELECT * FROM RapportDetectionIA WHERE soumission_id=?", (r["id"],)).fetchone()
        r["ia"] = dict(ia) if ia else None
        corr = db.execute("SELECT * FROM CorrectionAutomatique WHERE soumission_id=?", (r["id"],)).fetchone()
        r["correction"] = dict(corr) if corr else None
    return rows

@app.post("/soumissions")
def submit(body: SoumissionIn, u=Depends(current_user), db=Depends(get_db)):
    if u["role"] != "student": raise HTTPException(403)
    c = db.cursor()
    c.execute("INSERT INTO Soumission(devoir_id,etudiant_id,statut_global) VALUES(?,?,?)",
              (body.devoir_id, u["id"], "soumis"))
    s_id = c.lastrowid
    for r in body.reponses:
        c.execute("INSERT INTO ReponseEtudiant(soumission_id,question_id,texte_etudiant) VALUES(?,?,?)",
                  (s_id, r.question_id, r.texte))
    # Fake IA analysis
    ia_score = round(random.uniform(0.02, 0.95), 2)
    c.execute("INSERT INTO RapportDetectionIA(soumission_id,score_ia_global,statut_analyse,details_phrases_suspectes) VALUES(?,?,?,?)",
              (s_id, ia_score, "analysé", json.dumps([])))
    db.commit()
    return {"id": s_id, "ia_score": ia_score}

@app.post("/soumissions/{sid}/noter")
def noter(sid: int, body: GradeIn, u=Depends(current_user), db=Depends(get_db)):
    if u["role"] != "teacher": raise HTTPException(403)
    db.execute("UPDATE Soumission SET note_finale_humaine=?, statut_global='noté' WHERE id=?", (body.note, sid))
    db.execute("UPDATE CorrectionAutomatique SET statut_correction='validé' WHERE soumission_id=?", (sid,))
    db.commit()
    return {"ok": True}

@app.post("/soumissions/{sid}/analyser-ia")
def analyser_ia(sid: int, u=Depends(current_user), db=Depends(get_db)):
    if u["role"] != "teacher": raise HTTPException(403)
    score = round(random.uniform(0.01, 0.99), 2)
    phrases = ["L'intégrale est définie comme la limite..."] if score > 0.5 else []
    existing = db.execute("SELECT id FROM RapportDetectionIA WHERE soumission_id=?", (sid,)).fetchone()
    if existing:
        db.execute("UPDATE RapportDetectionIA SET score_ia_global=?,statut_analyse='analysé',details_phrases_suspectes=? WHERE soumission_id=?",
                   (score, json.dumps(phrases), sid))
    else:
        db.execute("INSERT INTO RapportDetectionIA(soumission_id,score_ia_global,statut_analyse,details_phrases_suspectes) VALUES(?,?,?,?)",
                   (sid, score, "analysé", json.dumps(phrases)))
    db.commit()
    return {"score_ia_global": score, "phrases": phrases}

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


def stats(u=Depends(current_user), db=Depends(get_db)):
    if u["role"] != "teacher": raise HTTPException(403)
    total = db.execute("SELECT COUNT(*) FROM Soumission").fetchone()[0]
    detected = db.execute("SELECT COUNT(*) FROM RapportDetectionIA WHERE score_ia_global > 0.5").fetchone()[0]
    avg_row = db.execute("SELECT AVG(note_finale_humaine) FROM Soumission WHERE note_finale_humaine IS NOT NULL").fetchone()[0]
    pending = db.execute("SELECT COUNT(*) FROM Soumission WHERE statut_global='soumis'").fetchone()[0]
    return {"total": total, "detected": detected, "moyenne": round(avg_row or 0, 1), "pending": pending}

if __name__ == "__main__":
    import uvicorn
    init_db()
    uvicorn.run(app, host="0.0.0.0", port=8000)
