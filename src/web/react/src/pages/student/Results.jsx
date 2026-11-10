import { useState, useEffect, useCallback } from 'react';
import { apiFetch } from '../../utils/api';
import { toast } from '../../utils/toast';

export default function StudentResults() {
  const [subs, setSubs]     = useState([]);
  const [devoirs, setDevoirs] = useState({});
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const [s, d] = await Promise.all([apiFetch('GET','/soumissions'), apiFetch('GET','/devoirs')]);
      setSubs(s||[]);
      const m={};(d||[]).forEach(x=>m[x.id]=x.titre);setDevoirs(m);
    } catch { toast('Erreur chargement'); }
    finally { setLoading(false); }
  }, []);

  useEffect(() => { load(); }, [load]);

  const noted = subs.filter(s => s.note_finale_humaine != null);
  const avg   = noted.length ? (noted.reduce((a,s)=>a+s.note_finale_humaine,0)/noted.length).toFixed(1) : '—';
  const best  = noted.length ? Math.max(...noted.map(s=>s.note_finale_humaine)) : '—';

  return (
    <div className="page-enter">
      <div style={{marginBottom:24}}>
        <h1 style={{fontSize:26,marginBottom:4}}>Mes résultats</h1>
        <p style={{color:'var(--mu)',fontSize:13}}>Vos notes et retours de correction.</p>
      </div>
      <div className="stats-row" style={{marginBottom:18}}>
        {[
          {label:'Moyenne générale', value:avg, sub:'/ 20'},
          {label:'Évals complètes', value:subs.length, sub:'soumissions'},
          {label:'Meilleure note', value:best, sub:'/ 20'},
        ].map(s=>(
          <div className="stat-card" key={s.label}>
            <div className="stat-label">{s.label}</div>
            <div className="stat-value">{s.value}</div>
            <div className="stat-sub">{s.sub}</div>
          </div>
        ))}
      </div>
      <div className="card">
        <div className="card-title">Détail par évaluation</div>
        {loading ? <p style={{color:'var(--mu)',fontSize:13}}>Chargement…</p>
          : subs.map(s=>{
            const note=s.note_finale_humaine;
            return (
              <div key={s.id} style={{
                display:'flex',alignItems:'center',gap:13,
                padding:13,border:'1px solid var(--bd)',
                borderRadius:'var(--r)',marginBottom:8,
              }}>
                {note!=null
                  ? <div className="grade-mark" aria-label={`Note : ${note} sur 20`}><span>{note}</span><small>/20</small></div>
                  : <div className="grade-mark" style={{ borderColor:'var(--bd)', color:'var(--mu)', transform:'none' }}><span>—</span></div>}
                <div style={{flex:1}}>
                  <div style={{fontSize:13,fontWeight:600,marginBottom:2}}>
                    {devoirs[s.devoir_id]||`Devoir #${s.devoir_id}`}
                  </div>
                  <div style={{fontSize:12,color:'var(--mu)'}}>Rendu le {s.date_depot?.slice(0,10)||'—'}</div>
                  {s.correction?.commentaires_ia && (
                    <div style={{fontSize:12,color:'var(--mu)',fontStyle:'italic',marginTop:2}}>
                      "{s.correction.commentaires_ia}"
                    </div>
                  )}
                </div>
                {note!=null
                  ? <span className={`badge ${note>=14?'badge-green':'badge-blue'}`}>
                      {note>=14?'Excellent':'Bien'}
                    </span>
                  : <span className="badge badge-gray">En correction</span>
                }
              </div>
            );
          })
        }
      </div>
    </div>
  );
}
