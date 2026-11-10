import { useState, useEffect, useCallback } from 'react';
import { apiFetch } from '../../utils/api';
import { toast } from '../../utils/toast';
import Modal from '../../components/Modal';
import IaBadge from '../../components/IaBadge';

// Note suggérée par la correction automatique, tant que l'enseignant ne l'a pas validée
function noteSuggeree(s) {
  const c = s.correction;
  return c && c.note_ia_suggeree != null && c.statut_correction !== 'validé' ? c.note_ia_suggeree : null;
}

// Surligne les phrases jugées IA (au-dessus du seuil du modèle) dans une réponse
function surligner(texte, suspectes) {
  const phrases = suspectes
    .map(p => ({ t: (p?.texte || '').trim(), proba: p?.probabilite_ia || 0 }))
    .filter(p => p.t.length >= 8 && texte.includes(p.t));
  if (!phrases.length) return texte;
  const re = new RegExp(`(${phrases.map(p => p.t.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|')})`);
  return texte.split(re).map((part, i) => {
    const p = phrases.find(x => x.t === part);
    return p
      ? <mark key={i} className="ia-suspect" title={`Probabilité IA : ${Math.round(p.proba * 100)} %`}>{part}</mark>
      : part;
  });
}

export default function Submissions() {
  const [subs, setSubs]           = useState([]);
  const [stats, setStats]         = useState(null);
  const [loading, setLoading]     = useState(true);
  const [current, setCurrent]     = useState(null);   // soumission ouverte
  const [reponses, setReponses]   = useState(null);   // null = chargement
  const [gradeOpen, setGradeOpen] = useState(false);
  const [gradeVal, setGradeVal]   = useState('');
  const [gradeSid, setGradeSid]   = useState(null);
  const [gradeNom, setGradeNom]   = useState('');

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const [s, st] = await Promise.all([
        apiFetch('GET', '/soumissions'),
        apiFetch('GET', '/stats'),
      ]);
      setSubs(s || []); setStats(st);
    } catch { toast('Erreur chargement'); }
    finally { setLoading(false); }
  }, []);

  useEffect(() => { load(); }, [load]);

  async function runIA(sid, e) {
    e.stopPropagation();
    const btn = e.currentTarget;
    btn.disabled = true; btn.textContent = 'Analyse…';
    try {
      const res = await apiFetch('POST', `/soumissions/${sid}/analyser-ia`);
      if (res.statut_analyse === 'analysé')
        toast(`Score IA : ${Math.round(res.score_ia_global * 100)} % (seuil ${Math.round(res.seuil * 100)} %)`);
      else toast(`Non analysé : ${res.message || res.statut_analyse}`);
      load();
    } catch { toast('Erreur analyse'); }
    finally { btn.disabled = false; btn.textContent = 'Analyser'; }
  }

  async function runCorrection(sid, e) {
    const btn = e.currentTarget;
    btn.disabled = true; btn.textContent = 'Correction…';
    try {
      const res = await apiFetch('POST', `/soumissions/${sid}/corriger-auto`);
      toast(`Note suggérée : ${res.note_ia_suggeree}/20 — ${res.commentaires_ia}`);
      load();
    } catch (err) {
      toast('' + (err?.detail || 'Erreur de correction'));
      btn.disabled = false; btn.textContent = 'Corriger';
    }
  }

  async function openCopie(s) {
    setCurrent(s); setReponses(null);
    try {
      const r = await apiFetch('GET', `/soumissions/${s.id}/reponses`);
      setReponses(r || []);
    } catch { setReponses([]); toast('Erreur lors du chargement de la copie'); }
  }

  // Pré-remplit avec la note suggérée par la correction automatique, si elle existe
  function openGrade(s) {
    const sugg = s.correction?.note_ia_suggeree;
    setGradeSid(s.id); setGradeNom(s.etudiant_nom);
    setGradeVal(sugg != null ? String(sugg) : ''); setGradeOpen(true);
  }

  let suspectes = [];
  try { suspectes = JSON.parse(current?.ia?.details_phrases_suspectes || '[]'); } catch { /* rapport illisible : pas de surlignage */ }

  async function submitGrade() {
    const note = parseFloat(gradeVal);
    if (isNaN(note) || note < 0 || note > 20) { toast('Note entre 0 et 20'); return; }
    try {
      await apiFetch('POST', `/soumissions/${gradeSid}/noter`, { note });
      toast(`Note ${note}/20 enregistrée`);
      setGradeOpen(false); load();
    } catch { toast('Erreur notation'); }
  }

  return (
    <div className="page-enter">
      <div style={{ marginBottom:24 }}>
        <h1 style={{  fontSize:26, letterSpacing:'-.02em', marginBottom:4 }}>
          Copies rendues
        </h1>
        <p style={{ color:'var(--mu)', fontSize:13 }}>Corrigez les copies et analysez l'usage de l'IA.</p>
      </div>

      {/* Stats */}
      {stats && (
        <div className="stats-row">
          {[
            { label:'Total copies', value:stats.total, sub:'soumissions' },
            { label:'Corrigées',    value:stats.total - stats.pending, sub:`sur ${stats.total}` },
            { label:'Alertes IA',   value:stats.detected, sub:'au-dessus du seuil du modèle' },
            { label:'Moyenne',      value:stats.moyenne || '—', sub:'/ 20' },
          ].map(s => (
            <div className="stat-card" key={s.label}>
              <div className="stat-label">{s.label}</div>
              <div className="stat-value">{s.value}</div>
              <div className="stat-sub">{s.sub}</div>
            </div>
          ))}
        </div>
      )}

      {/* Tableau */}
      {loading ? (
        <p style={{ color:'var(--mu)', fontSize:13 }}>Chargement…</p>
      ) : (
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Étudiant</th>
                <th>Devoir</th>
                <th>Rendu le</th>
                <th>Statut</th>
                <th>Note</th>
                <th>Score IA</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {subs.length === 0 ? (
                <tr><td colSpan={7} style={{ textAlign:'center', color:'var(--mu)' }}>Aucune copie</td></tr>
              ) : subs.map(s => {
                const noted = s.statut_global === 'noté';
                const sugg  = noteSuggeree(s);
                return (
                  <tr key={s.id} style={{ cursor:'pointer' }} onClick={() => openCopie(s)}>
                    <td><strong>{s.etudiant_nom}</strong></td>
                    <td style={{ color:'var(--mu)', fontSize:12 }}>Devoir #{s.devoir_id}</td>
                    <td style={{  fontSize:12 }}>
                      {s.date_depot?.slice(0, 10) || '—'}
                    </td>
                    <td>
                      {noted
                        ? <span className="badge badge-low">Notée</span>
                        : <span className="badge badge-check">À noter</span>}
                    </td>
                    <td style={{ fontWeight:600 }}>
                      {s.note_finale_humaine != null
                        ? `${s.note_finale_humaine}/20`
                        : sugg != null
                          ? <span style={{ color:'var(--mu)', fontWeight:400 }}
                              title="Note suggérée par la correction automatique">~{sugg}/20</span>
                          : '—'}
                    </td>
                    <td><IaBadge ia={s.ia} /></td>
                    <td onClick={e => e.stopPropagation()}>
                      <div style={{ display:'flex', gap:5, flexWrap:'wrap' }}>
                        <button className="btn btn-secondary btn-sm" onClick={() => openCopie(s)}>
                          Voir
                        </button>
                        {!noted && (
                          <button className="btn btn-secondary btn-sm" onClick={e => runCorrection(s.id, e)}
                            title="Correction automatique (note suggérée)">
                            Corriger
                          </button>
                        )}
                        {!noted && (
                          <button className="btn btn-secondary btn-sm" onClick={() => openGrade(s)}>
                            Noter
                          </button>
                        )}
                        <button className="btn btn-dark btn-sm" onClick={e => runIA(s.id, e)}>
                          Analyser
                        </button>
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {/* Modal copie */}
      <Modal
        open={!!current}
        onClose={() => setCurrent(null)}
        title={current ? `Copie de ${current.etudiant_nom}` : ''}
        subtitle={current ? `Devoir #${current.devoir_id} · ${current.date_depot?.slice(0,10)}`
          + (current.note_finale_humaine != null ? ` · Note : ${current.note_finale_humaine}/20` : '') : ''}
        actions={<>
          <button className="btn btn-secondary" onClick={() => setCurrent(null)}>Fermer</button>
          {current && current.statut_global !== 'noté' && (
            <button className="btn btn-primary"
              onClick={() => { setCurrent(null); openGrade(current); }}>
              Attribuer une note
            </button>
          )}
        </>}
      >
        {current?.ia && <div style={{ marginBottom:12 }}><IaBadge ia={current.ia} prefix="IA " /></div>}
        {reponses === null
          ? <p style={{ color:'var(--mu)', fontSize:13 }}>Chargement des réponses…</p>
          : reponses.length === 0
          ? <p style={{ color:'var(--mu)', fontSize:13 }}>Aucune réponse enregistrée.</p>
          : reponses.map((r, i) => (
            <div key={i} style={{
              background:'var(--grl)', border:'1px solid #D9CEF7',
              borderRadius:8, padding:13, marginBottom:10,
            }}>
              <div style={{ fontSize:11, fontWeight:600, color:'var(--gr)', marginBottom:4 }}>
                Q{r.numero_ordre} — {r.points_attribues} pt(s)
              </div>
              <div style={{ fontSize:13, marginBottom:7 }}>{r.enonce}</div>
              <div style={{
                background:'var(--sf)', border:'1px solid var(--bd)',
                borderRadius:6, padding:10, fontSize:13, lineHeight:1.7,
              }}>
                {r.texte_etudiant
                  ? surligner(r.texte_etudiant, suspectes)
                  : <em style={{ color:'var(--mu)' }}>Pas de réponse</em>}
              </div>
            </div>
          ))
        }
      </Modal>

      {/* Modal notation */}
      <Modal
        open={gradeOpen}
        onClose={() => setGradeOpen(false)}
        title={`Noter ${gradeNom}`}
        actions={<>
          <button className="btn btn-secondary" onClick={() => setGradeOpen(false)}>Annuler</button>
          <button className="btn btn-primary" onClick={submitGrade}>Enregistrer</button>
        </>}
      >
        <div className="fg">
          <label htmlFor="gradeVal">Note (sur 20)</label>
          <input id="gradeVal" type="number" min={0} max={20} step={0.5}
            value={gradeVal} onChange={e => setGradeVal(e.target.value)}
            placeholder="ex: 14.5" />
        </div>
      </Modal>
    </div>
  );
}
