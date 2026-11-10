export default function StudentCourses() {
  return (
    <div className="page-enter">
      <div style={{marginBottom:24}}>
        <h1 style={{fontSize:26,marginBottom:4}}>Mes cours</h1>
        <p style={{color:'var(--mu)',fontSize:13}}>Les supports de cours partagés par vos professeurs.</p>
      </div>
      <div className="card" style={{textAlign:'center',padding:'40px 20px'}}>
        <div style={{fontWeight:600,marginBottom:6}}>Aucun cours disponible</div>
        <p style={{color:'var(--mu)',fontSize:13}}>
          Vos professeurs n'ont pas encore partagé de supports de cours.
        </p>
      </div>
    </div>
  );
}
