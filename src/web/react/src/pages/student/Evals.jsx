import { useState, useEffect, useCallback } from 'react';
import { apiFetch } from '../../utils/api';
import { toast } from '../../utils/toast';
import Modal from '../../components/Modal';
import { initialesMatiere } from '../../utils/matieres';

export default function StudentEvals() {
  const [devoirs, setDevoirs]   = useState([]);
  const [doneIds, setDoneIds]   = useState(new Set());
  const [loading, setLoading]   = useState(true);
  const [examOpen, setExamOpen] = useState(false);
  const [examDevoir, setExamDevoir] = useState(null);
  const [examQs, setExamQs]     = useState([]);
  const [reponses, setReponses] = useState({});
  const [submitting, setSubmitting] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const [d, s] = await Promise.all([
        apiFetch('GET', '/devoirs'),
        apiFetch('GET', '/soumissions'),
      ]);
      setDevoirs(d || []);
      setDoneIds(new Set((s||[]).map(x => x.devoir_id)));
    } catch { toast('Erreur chargement'); }
    finally { setLoading(false); }
  }, []);

  useEffect(() => { load(); }, [load]);

  async function startExam(d) {
    setExamDevoir(d); setReponses({});
    setExamOpen(true);
    try {
      const qs = await apiFetch('GET', `/devoirs/${d.id}/questions`);
      setExamQs(qs || []);
    } catch { setExamQs([]); toast('Erreur chargement questions'); }
  }

  async function submit() {
    setSubmitting(true);
    try {
      const rep = examQs.map(q => ({ question_id:q.id, texte:reponses[q.id]||'' }));
      const res = await apiFetch('POST', '/soumissions', { devoir_id:examDevoir.id, reponses:rep });
      toast(`Copie rendue ! Score IA : ${Math.round(res.ia_score*100)}%`);
      setExamOpen(false); load();
    } catch (e) { toast(''+(e.detail||'Erreur soumission')); }
    finally { setSubmitting(false); }
  }

  const pending = devoirs.filter(d => !doneIds.has(d.id));
  const done    = devoirs.filter(d =>  doneIds.has(d.id));

  return (
    <div className="page-enter">
      <div style={{ marginBottom:24 }}>
        <h1 style={{  fontSize:26, marginBottom:4 }}>Mes évaluations</h1>
        <p style={{ color:'var(--mu)', fontSize:13 }}>Répondez aux évaluations de vos professeurs.</p>
      </div>

      {loading ? <p style={{color:'var(--mu)'}}>Chargement…</p> : <>
        <div style={{ fontSize:11, fontWeight:600, color:'var(--mu)',  marginBottom:8 }}>À rendre</div>
        {pending.length === 0
          ? <p style={{color:'var(--mu)',fontSize:13,marginBottom:18}}>Aucune évaluation en attente.</p>
          : pending.map(d => (
            <div key={d.id} className="card" style={{display:'flex',alignItems:'center',gap:13}}>
              <div aria-hidden="true" style={{width:44,height:44,borderRadius:8,background:'var(--sf2)',display:'flex',alignItems:'center',justifyContent:'center',fontWeight:800,fontSize:15,flexShrink:0}}>{initialesMatiere(d.titre)}</div>
              <div style={{flex:1}}>
                <div style={{fontWeight:600,fontSize:14}}>{d.titre}</div>
                <div style={{fontSize:12,color:'var(--mu)'}}>{d.description}</div>
                <div style={{fontSize:12,color:'var(--mu)',marginTop:3}}>
                  {d.questions?.length||0} question(s)
                  {d.date_limite && <> · <strong style={{color:'var(--or)'}}>Limite : {d.date_limite.slice(0,10)}</strong></>}
                </div>
              </div>
              <button className="btn btn-primary" onClick={() => startExam(d)}>Commencer</button>
            </div>
          ))
        }

        <div style={{ fontSize:11, fontWeight:600, color:'var(--mu)',  marginBottom:8, marginTop:18 }}>Rendus</div>
        {done.length === 0
          ? <p style={{color:'var(--mu)',fontSize:13}}>Aucun devoir rendu.</p>
          : done.map(d => (
            <div key={d.id} className="card" style={{display:'flex',alignItems:'center',gap:13}}>
              <div aria-hidden="true" style={{width:44,height:44,borderRadius:8,background:'var(--sf2)',display:'flex',alignItems:'center',justifyContent:'center',fontWeight:800,fontSize:15,flexShrink:0}}>{initialesMatiere(d.titre)}</div>
              <div style={{flex:1}}>
                <div style={{fontWeight:600,fontSize:14}}>{d.titre}</div>
                <div style={{fontSize:12,color:'var(--mu)'}}>{d.description}</div>
              </div>
              <span className="badge badge-green">Rendu</span>
            </div>
          ))
        }
      </>}

      {/* Modal examen */}
      <Modal open={examOpen} onClose={() => setExamOpen(false)}
        title={examDevoir?.titre || 'Évaluation'}
        subtitle={examDevoir?.description}
        actions={<>
          <button className="btn btn-secondary" onClick={() => setExamOpen(false)}>Annuler</button>
          <button className="btn btn-primary" onClick={submit} disabled={submitting}>
            {submitting ? 'Envoi…' : 'Rendre la copie'}
          </button>
        </>}
      >
        {examQs.map((q, i) => (
          <div key={q.id} style={{
            background:'var(--grl)', border:'1px solid #D9CEF7',
            borderRadius:8, padding:13, marginBottom:12,
          }}>
            <div style={{fontSize:11,fontWeight:600,color:'var(--gr)',marginBottom:5}}>
              Question {i+1} · {q.points_attribues} pt(s)
            </div>
            <div style={{fontSize:13,marginBottom:9}}>{q.enonce}</div>
            <textarea
              placeholder="Rédigez votre réponse…"
              value={reponses[q.id]||''}
              onChange={e => setReponses(prev => ({...prev,[q.id]:e.target.value}))}
              aria-label={`Réponse question ${i+1}`}
              style={{minHeight:80}}
            />
          </div>
        ))}
      </Modal>
    </div>
  );
}
