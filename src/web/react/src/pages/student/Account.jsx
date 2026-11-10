import { useState } from 'react';
import { toast } from '../../utils/toast';

export default function StudentAccount({ me }) {
  const [prenom, setPrenom] = useState(me?.nom?.split(' ')[0]||'');
  const [nom, setNom]       = useState(me?.nom?.split(' ')[1]||'');
  function save(e) { e.preventDefault(); toast('Profil mis à jour'); }
  return (
    <div className="page-enter">
      <div style={{marginBottom:24}}>
        <h1 style={{fontSize:26,marginBottom:4}}>Mon compte</h1>
        <p style={{color:'var(--mu)',fontSize:13}}>Gérez votre profil étudiant.</p>
      </div>
      <div className="card" style={{maxWidth:560}}>
        <div className="card-title">Informations personnelles</div>
        <form onSubmit={save}>
          <div className="form-row">
            <div className="fg">
              <label htmlFor="sPre">Prénom</label>
              <input id="sPre" value={prenom} onChange={e=>setPrenom(e.target.value)} />
            </div>
            <div className="fg">
              <label htmlFor="sNo">Nom</label>
              <input id="sNo" value={nom} onChange={e=>setNom(e.target.value)} />
            </div>
          </div>
          <button type="submit" className="btn btn-primary">Enregistrer</button>
        </form>
      </div>
    </div>
  );
}
