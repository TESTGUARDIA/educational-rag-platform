import { useState, useRef } from 'react';
import { apiFetch, apiUpload } from '../../utils/api';
import { toast } from '../../utils/toast';
import Modal from '../../components/Modal';

export default function Upload() {
  const [dragging, setDragging]   = useState(false);
  const [files, setFiles]         = useState([]);
  const [uploading, setUploading] = useState(false);
  const [showGenModal, setShowGenModal] = useState(false);
  const [nbQ, setNbQ]             = useState(5);
  const [matiere, setMatiere]     = useState('');
  const fileRef                   = useRef();

  function handleFiles(fileList) {
    const pdfs = Array.from(fileList).filter(f =>
      f.type === 'application/pdf' || f.name.toLowerCase().endsWith('.pdf')
    );
    if (pdfs.length !== fileList.length) toast('Seuls les fichiers PDF sont acceptés');
    setFiles(pdfs.slice(0, 10));
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
      setFiles([]);
    } catch (e) {
      toast('' + (e.detail || 'Erreur upload'));
    } finally {
      setUploading(false);
    }
  }

  async function generer() {
    setShowGenModal(false);
    toast('Génération en cours…');
    try {
      await apiFetch('POST', '/devoirs/1/generer', {
        document_id: 1, nb_questions: nbQ, matiere,
      });
      toast('Évaluation générée !');
    } catch (e) {
      toast('' + (e.detail || 'Erreur génération'));
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
                  onClick={() => setFiles(files.filter((_, j) => j !== i))}
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
          À partir d'un cours uploadé, générez des questions automatiquement.
        </div>
        <button className="btn btn-primary" onClick={() => setShowGenModal(true)}>
          Générer une évaluation
        </button>
      </div>

      {/* Modal génération */}
      <Modal
        open={showGenModal}
        onClose={() => setShowGenModal(false)}
        title="Générer une évaluation"
        subtitle="Paramétrez la génération automatique de questions"
        actions={<>
          <button className="btn btn-secondary" onClick={() => setShowGenModal(false)}>Annuler</button>
          <button className="btn btn-primary" onClick={generer}>Générer</button>
        </>}
      >
        <div className="fg">
          <label htmlFor="genNbQ">Nombre de questions</label>
          <input id="genNbQ" type="number" min={1} max={20} value={nbQ}
            onChange={e => setNbQ(+e.target.value)} />
        </div>
        <div className="fg">
          <label htmlFor="genMat">Matière (optionnel)</label>
          <input id="genMat" type="text" value={matiere}
            onChange={e => setMatiere(e.target.value)} placeholder="Mathématiques…" />
        </div>
        <p style={{ fontSize:11, color:'var(--mu)', fontStyle:'italic' }}>
          Branchement pôle Génération en cours (J3).
        </p>
      </Modal>
    </div>
  );
}
