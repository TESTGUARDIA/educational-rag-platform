import { useState, useRef } from 'react';
import { apiFetch, apiUpload } from '../../utils/api';
import { toast } from '../../utils/toast';

const NB_QUESTIONS_MAX = 30;   // même borne que l'API (main.py)

export default function Upload() {
  const [dragging, setDragging]   = useState(false);
  const [files, setFiles]         = useState([]);
  const [uploading, setUploading] = useState(false);
  const [genIdx, setGenIdx]       = useState(0);     // PDF utilisé pour la génération
  const [nbQ, setNbQ]             = useState(5);
  const [matiere, setMatiere]     = useState('');
  const [classe, setClasse]       = useState('');
  const [generating, setGenerating] = useState(false);
  const [genResult, setGenResult] = useState(null);  // { titre, nb_generees, questions }
  const [genError, setGenError]   = useState('');
  const fileRef                   = useRef();

  function handleFiles(fileList) {
    const pdfs = Array.from(fileList).filter(f =>
      f.type === 'application/pdf' || f.name.toLowerCase().endsWith('.pdf')
    );
    if (pdfs.length !== fileList.length) toast('Seuls les fichiers PDF sont acceptés');
    setFiles(pdfs.slice(0, 10));
    setGenIdx(0);
  }

  function removeFile(i) {
    setFiles(files.filter((_, j) => j !== i));
    setGenIdx(0);
  }

  async function uploadFiles() {
    if (!files.length) return;
    setUploading(true);
    try {
      const fd = new FormData();
      files.forEach(f => fd.append('files', f));
      const res = await apiUpload('/documents', fd);
      toast(`${res.sauvegardes.length} fichier(s) uploadé(s)`);
      if (res.erreurs?.length) toast(`${res.erreurs.length} erreur(s)`);
      // La liste est conservée : on peut générer une évaluation depuis ces cours
    } catch (e) {
      toast('' + (e.detail || 'Erreur upload'));
    } finally {
      setUploading(false);
    }
  }

  // Génération réelle (pôle Génération) : crée le devoir puis génère ses questions depuis le PDF
  async function generer() {
    if (generating) return;
    const pdf = files[genIdx];
    if (!pdf) { toast('Choisissez un cours au format PDF'); return; }
    const nb = Math.min(Math.max(parseInt(nbQ, 10) || 5, 1), NB_QUESTIONS_MAX);
    const titre = `${matiere.trim() || 'Évaluation'} — ${pdf.name.replace(/\.pdf$/i, '')}`;
    setGenerating(true); setGenResult(null); setGenError('');
    try {
      const dv = await apiFetch('POST', '/devoirs', {
        titre,
        description: `${classe.trim() ? `Classe cible : ${classe.trim()} · ` : ''}généré depuis ${pdf.name}`,
      });
      const fd = new FormData();
      fd.append('fichier', pdf);
      fd.append('nb_questions', String(nb));
      const res = await apiFetch('POST', `/devoirs/${dv.id}/generer`, fd);
      setGenResult({ titre, ...res });
      toast('Évaluation générée !');
    } catch (e) {
      setGenError(e?.detail || 'Erreur lors de la génération');
    } finally {
      setGenerating(false);
    }
  }

  return (
    <div className="page-enter">
      {/* Page header */}
      <div style={{ marginBottom:24 }}>
        <h1 style={{  fontSize:26, letterSpacing:'-.02em', marginBottom:4 }}>
          Cours &amp; Évaluations
        </h1>
        <p style={{ color:'var(--mu)', fontSize:13 }}>
          Déposez vos supports de cours, puis générez des évaluations automatiquement.
        </p>
      </div>

      {/* Upload zone */}
      <div className="card">
        <div className="card-title">Déposer des cours (PDF)</div>
        <div className="card-subtitle">Maximum 10 fichiers · 20 Mo chacun</div>

        <div
          className={`upload-zone${dragging ? ' drag-over' : ''}`}
          onDragOver={e => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={e => { e.preventDefault(); setDragging(false); handleFiles(e.dataTransfer.files); }}
          onClick={() => fileRef.current.click()}
          role="button"
          tabIndex={0}
          aria-label="Zone de dépôt de fichiers PDF"
          onKeyDown={e => e.key === 'Enter' && fileRef.current.click()}
        >
          <div style={{ fontWeight:600, marginBottom:4, fontSize:14 }}>
            Glisser-déposer des fichiers PDF
          </div>
          <div style={{ color:'var(--mu)', fontSize:12 }}>
            ou <span style={{ color:'var(--gr)', fontWeight:600 }}>parcourir</span>
          </div>
        </div>

        {/* Bouton fallback obligatoire WCAG 2.2 SC 2.5.7 */}
        <input
          ref={fileRef}
          type="file"
          accept=".pdf,application/pdf"
          multiple
          style={{ display:'none' }}
          aria-label="Sélectionner des fichiers PDF"
          onChange={e => handleFiles(e.target.files)}
        />
        <button
          className="btn btn-secondary"
          style={{ marginTop:10 }}
          onClick={() => fileRef.current.click()}
        >
          Parcourir les fichiers
        </button>

        {/* Liste des fichiers sélectionnés */}
        {files.length > 0 && (
          <div style={{ marginTop:16 }}>
            {files.map((f, i) => (
              <div key={i} style={{
                display:'flex', alignItems:'center', gap:10,
                padding:'8px 12px', background:'var(--bg)',
                border:'1px solid var(--bd)', borderRadius:6, marginBottom:6,
              }}>
                <span style={{ fontSize:16 }}></span>
                <span style={{ flex:1, fontSize:13 }}>{f.name}</span>
                <span style={{ fontSize:11, color:'var(--mu)',  }}>
                  {(f.size / 1024 / 1024).toFixed(1)} Mo
                </span>
                <button
                  className="btn btn-danger btn-sm"
                  onClick={() => removeFile(i)}
                  aria-label={`Supprimer ${f.name}`}
                >✕</button>
              </div>
            ))}
            <div style={{ display:'flex', gap:9, marginTop:12 }}>
              <button className="btn btn-primary" onClick={uploadFiles} disabled={uploading}>
                {uploading ? 'Upload…' : `Uploader ${files.length} fichier(s)`}
              </button>
              <button className="btn btn-secondary" onClick={() => setFiles([])}>
                Annuler
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Génération */}
      <div className="card">
        <div className="card-title">Générer une évaluation</div>
        <div className="card-subtitle">
          Choisissez un cours déposé ci-dessus : un devoir est créé et ses questions sont générées.
        </div>

        {files.length > 1 && (
          <div className="fg">
            <label htmlFor="genFile">Cours utilisé</label>
            <select id="genFile" value={genIdx} onChange={e => setGenIdx(+e.target.value)}>
              {files.map((f, i) => <option key={i} value={i}>{f.name}</option>)}
            </select>
          </div>
        )}
        <div style={{ display:'grid', gridTemplateColumns:'repeat(auto-fit, minmax(160px, 1fr))', gap:13 }}>
          <div className="fg">
            <label htmlFor="genMat">Matière</label>
            <input id="genMat" type="text" value={matiere}
              onChange={e => setMatiere(e.target.value)} placeholder="Mathématiques…" />
          </div>
          <div className="fg">
            <label htmlFor="genClasse">Classe cible</label>
            <input id="genClasse" type="text" value={classe}
              onChange={e => setClasse(e.target.value)} placeholder="Terminale A…" />
          </div>
          <div className="fg">
            <label htmlFor="genNbQ">Questions</label>
            <input id="genNbQ" type="number" min={1} max={NB_QUESTIONS_MAX} value={nbQ}
              onChange={e => setNbQ(e.target.value)} />
          </div>
        </div>
        <button className="btn btn-primary" onClick={generer} disabled={generating || !files.length}>
          {generating ? 'Génération…' : 'Générer une évaluation'}
        </button>
        {!files.length && (
          <p style={{ fontSize:12, color:'var(--mu)', marginTop:8 }}>Déposez d'abord un cours au format PDF.</p>
        )}

        <div aria-live="polite" style={{ marginTop:16 }}>
          {generating && (
            <p style={{ color:'var(--mu)', fontSize:13 }}>
              Génération en cours… (de quelques secondes à quelques minutes selon le cours)
            </p>
          )}
          {genError && <p style={{ fontSize:13, fontWeight:600 }}>{genError}</p>}
          {genResult && <>
            <div className="card-title" style={{ fontSize:14, marginBottom:8 }}>
              {genResult.nb_generees} question(s) générée(s) — devoir « {genResult.titre} »
            </div>
            {genResult.questions.map(q => (
              <div key={q.id} style={{
                border:'1px solid var(--bd)', borderRadius:8, padding:11,
                marginBottom:8, background:'var(--bg)',
              }}>
                <div style={{ fontSize:13, fontWeight:600, marginBottom:4 }}>Q{q.numero_ordre}. {q.enonce}</div>
                <div style={{ fontSize:12, color:'var(--mu)' }}>Réponse attendue : {q.reponse_attendue}</div>
              </div>
            ))}
          </>}
        </div>
      </div>
    </div>
  );
}
