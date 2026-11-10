import { useState, useRef } from 'react';
import { toast } from '../../utils/toast';
import Modal from '../../components/Modal';

const INIT_STUDENTS = [
  {p:'Lucas',n:'Moreau',e:'lucas.moreau@lycee.fr',c:'Terminale A',ev:3,my:'14.5'},
  {p:'Emma',n:'Lefebvre',e:'emma.lefebvre@lycee.fr',c:'Terminale A',ev:2,my:'16'},
  {p:'Hugo',n:'Bernard',e:'hugo.bernard@lycee.fr',c:'Terminale B',ev:1,my:'—'},
  {p:'Chloé',n:'Martin',e:'chloe.martin@lycee.fr',c:'Terminale B',ev:2,my:'12'},
];

export default function Students() {
  const [students, setStudents] = useState(INIT_STUDENTS);
  const [search, setSearch]     = useState('');
  const [filterC, setFilterC]   = useState('');
  const [addOpen, setAddOpen]   = useState(false);
  const [csvOpen, setCsvOpen]   = useState(false);
  const [csvRows, setCsvRows]   = useState([]);
  const [csvPreview, setCsvPreview] = useState(null);
  const [newS, setNewS]         = useState({ p:'', n:'', e:'', c:'Terminale A' });
  const csvRef = useRef();

  const classes = [...new Set(students.map(s => s.c))];
  const filtered = students.filter(s =>
    (!search || `${s.p} ${s.n} ${s.e}`.toLowerCase().includes(search.toLowerCase())) &&
    (!filterC || s.c === filterC)
  );

  function addStudent() {
    if (!newS.p || !newS.n || !newS.e) { toast('Remplissez tous les champs'); return; }
    setStudents(prev => [...prev, { ...newS, ev:0, my:'—' }]);
    setAddOpen(false); setNewS({ p:'', n:'', e:'', c:'Terminale A' });
    toast(`${newS.p} ${newS.n} ajouté(e)`);
  }

  function delStudent(i) {
    const s = filtered[i];
    const idx = students.indexOf(s);
    const copy = [...students]; copy.splice(idx, 1); setStudents(copy);
    toast(`${s.p} ${s.n} supprimé(e)`);
  }

  function parseCSV(file) {
    if (!file) return;
    const r = new FileReader();
    r.onload = ev => {
      const lines = ev.target.result.trim().split('\n');
      const hdr = lines[0].toLowerCase().replace(/\r/g,'').split(',').map(h=>h.trim());
      const iP=hdr.indexOf('prenom'),iN=hdr.indexOf('nom'),iE=hdr.indexOf('email'),iC=hdr.indexOf('classe');
      if (iP<0||iN<0||iE<0) { toast('Colonnes requises : prenom, nom, email'); return; }
      const rows = lines.slice(1).filter(l=>l.trim()).map(line => {
        const c = line.replace(/\r/g,'').split(',').map(x=>x.trim());
        return { p:c[iP]||'', n:c[iN]||'', e:c[iE]||'', c:iC>=0?c[iC]||'':'' };
      });
      const valid = rows.filter(r => r.p && r.n && r.e && r.e.includes('@'));
      setCsvRows(valid); setCsvPreview({ total:rows.length, valid:valid.length, name:file.name });
    };
    r.readAsText(file, 'UTF-8');
  }

  function importCSV() {
    let added = 0;
    const existing = new Set(students.map(s=>s.e));
    const toAdd = csvRows.filter(r => !existing.has(r.e)).map(r => ({...r, ev:0, my:'—'}));
    setStudents(prev => [...prev, ...toAdd]);
    added = toAdd.length;
    setCsvOpen(false); setCsvRows([]); setCsvPreview(null);
    toast(`${added} élève(s) importé(s)`);
  }

  function dlTemplate() {
    const csv = 'prenom,nom,email,classe\nThomas,Leclerc,thomas.leclerc@lycee.fr,Terminale A\n';
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([csv], {type:'text/csv'}));
    a.download = 'modele_eleves.csv'; a.click();
    toast('Modèle CSV téléchargé');
  }

  return (
    <div className="page-enter">
      <div style={{ marginBottom:24 }}>
        <h1 style={{  fontSize:26, letterSpacing:'-.02em', marginBottom:4 }}>
          Élèves
        </h1>
        <p style={{ color:'var(--mu)', fontSize:13 }}>
          {filtered.length} élève(s) · {classes.length} classe(s)
        </p>
      </div>

      {/* Barre d'actions */}
      <div style={{ display:'flex', gap:9, marginBottom:16, flexWrap:'wrap' }}>
        <input
          type="search" placeholder="Rechercher…"
          value={search} onChange={e => setSearch(e.target.value)}
          style={{ maxWidth:220 }}
          aria-label="Rechercher un élève"
        />
        <select value={filterC} onChange={e => setFilterC(e.target.value)}
          style={{ maxWidth:160 }} aria-label="Filtrer par classe">
          <option value="">Toutes les classes</option>
          {classes.map(c => <option key={c}>{c}</option>)}
        </select>
        <button className="btn btn-primary" onClick={() => setAddOpen(true)}>+ Ajouter</button>
        <button className="btn btn-secondary" onClick={() => setCsvOpen(true)}>Importer CSV</button>
        <button className="btn btn-secondary" onClick={dlTemplate}>Modèle CSV</button>
      </div>

      {/* Tableau */}
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Nom</th><th>Email</th><th>Classe</th>
              <th style={{textAlign:'center'}}>Évals</th>
              <th>Moyenne</th><th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {filtered.length === 0
              ? <tr><td colSpan={6} style={{textAlign:'center',color:'var(--mu)'}}>Aucun élève</td></tr>
              : filtered.map((s, i) => (
                <tr key={i}>
                  <td><strong>{s.p} {s.n}</strong></td>
                  <td style={{fontSize:12}}>{s.e}</td>
                  <td><span className="badge badge-gray">{s.c}</span></td>
                  <td style={{textAlign:'center',}}>{s.ev}</td>
                  <td style={{fontWeight:600}}>
                    {s.my !== '—' ? `${s.my}/20` : '—'}
                  </td>
                  <td>
                    <button className="btn btn-danger btn-sm"
                      onClick={() => delStudent(i)}
                      aria-label={`Supprimer ${s.p} ${s.n}`}></button>
                  </td>
                </tr>
              ))
            }
          </tbody>
        </table>
      </div>

      {/* Modal ajout */}
      <Modal open={addOpen} onClose={() => setAddOpen(false)}
        title="Ajouter un élève"
        actions={<>
          <button className="btn btn-secondary" onClick={() => setAddOpen(false)}>Annuler</button>
          <button className="btn btn-primary" onClick={addStudent}>Ajouter</button>
        </>}
      >
        <div className="form-row">
          <div className="fg">
            <label htmlFor="addPrenom">Prénom</label>
            <input id="addPrenom" value={newS.p} onChange={e=>setNewS({...newS,p:e.target.value})} />
          </div>
          <div className="fg">
            <label htmlFor="addNom">Nom</label>
            <input id="addNom" value={newS.n} onChange={e=>setNewS({...newS,n:e.target.value})} />
          </div>
        </div>
        <div className="fg">
          <label htmlFor="addEmail">Email</label>
          <input id="addEmail" type="email" value={newS.e} onChange={e=>setNewS({...newS,e:e.target.value})} />
        </div>
        <div className="fg">
          <label htmlFor="addClasse">Classe</label>
          <input id="addClasse" value={newS.c} onChange={e=>setNewS({...newS,c:e.target.value})} />
        </div>
      </Modal>

      {/* Modal CSV */}
      <Modal open={csvOpen} onClose={() => setCsvOpen(false)}
        title="Importer des élèves (CSV)"
        subtitle="Colonnes requises : prenom, nom, email — optionnel : classe"
        actions={<>
          <button className="btn btn-secondary" onClick={() => setCsvOpen(false)}>Annuler</button>
          {csvPreview && <button className="btn btn-primary" onClick={importCSV}>
            Importer {csvPreview.valid} élève(s)
          </button>}
        </>}
      >
        <div
          className="upload-zone"
          onClick={() => csvRef.current.click()}
          onDragOver={e => e.preventDefault()}
          onDrop={e => { e.preventDefault(); parseCSV(e.dataTransfer.files[0]); }}
          role="button" tabIndex={0}
          onKeyDown={e => e.key==='Enter' && csvRef.current.click()}
          aria-label="Déposer un fichier CSV"
          style={{ padding:20 }}
        >
          Glisser ou cliquer pour choisir un CSV
        </div>
        <input ref={csvRef} type="file" accept=".csv" style={{display:'none'}}
          aria-label="Sélectionner un fichier CSV"
          onChange={e => parseCSV(e.target.files[0])} />
        {csvPreview && (
          <p style={{ fontSize:12, color:'var(--mu)', marginTop:10 }}>
            {csvPreview.name} · {csvPreview.total} lignes · {csvPreview.valid} valides
          </p>
        )}
      </Modal>
    </div>
  );
}
