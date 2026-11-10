import Logo from './Logo';

const PAGES_T = [
  { id: 'upload',   label: 'Cours et évaluations' },
  { id: 'subs',     label: 'Copies' },
  { id: 'students', label: 'Élèves' },
  { id: 'account',  label: 'Compte' },
];
const PAGES_S = [
  { id: 'evals',    label: 'Évaluations' },
  { id: 'results',  label: 'Résultats' },
  { id: 'courses',  label: 'Cours' },
  { id: 'account',  label: 'Compte' },
];

export default function Header({ me, page, setPage, onLogout }) {
  const isTeacher = me?.role === 'teacher';
  const pages = isTeacher ? PAGES_T : PAGES_S;
  const initials = (me?.nom || '').split(' ').map(w => w[0]).join('').slice(0, 2).toUpperCase();

  return (
    <header style={{
      background: 'var(--sf)', borderBottom: '1px solid var(--bd)',
      position: 'sticky', top: 0, zIndex: 100,
    }}>
      <a href="#main-content" className="skip-link" style={{
        position: 'absolute', top: '-40px', left: 0, background: 'var(--gr)', color: 'var(--sf)',
        padding: '8px 16px', zIndex: 1000, borderRadius: '0 0 6px 0', fontSize: 13, fontWeight: 600,
      }}
        onFocus={e => { e.target.style.top = '0'; }}
        onBlur={e => { e.target.style.top = '-40px'; }}
      >
        Aller au contenu principal
      </a>

      <div style={{
        maxWidth: 1200, margin: '0 auto', padding: '0 20px',
        display: 'flex', alignItems: 'center', gap: 24, minHeight: 64, flexWrap: 'wrap',
      }}>
        <Logo size={24} />

        <nav style={{ display: 'flex', gap: isTeacher ? 20 : 6, flex: 1, flexWrap: 'wrap' }} aria-label="Navigation principale">
          {pages.map(p => {
            const active = page === p.id;
            // Enseignant : soulignement violet, sobre. Élève : pastille pleine, plus accueillante.
            const style = isTeacher
              ? { color: active ? 'var(--tx)' : 'var(--mu)', borderBottom: active ? '2px solid var(--gr)' : '2px solid transparent', padding: '20px 0', borderRadius: 0 }
              : { color: active ? 'var(--sf)' : 'var(--tx)', background: active ? 'var(--tx)' : 'transparent', padding: '9px 14px', borderRadius: 6 };
            return (
              <button
                key={p.id}
                onClick={() => setPage(p.id)}
                aria-current={active ? 'page' : undefined}
                style={{
                  background: 'transparent', border: 'none', cursor: 'pointer',
                  fontFamily: 'inherit', fontSize: 14, fontWeight: active ? 600 : 500, ...style,
                }}
              >
                {p.label}
              </button>
            );
          })}
        </nav>

        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <span style={{ color: 'var(--mu)', fontSize: 13 }}>{me?.nom}</span>
          <div aria-hidden="true" style={{
            width: 34, height: 34, borderRadius: '50%',
            background: isTeacher ? 'var(--gr)' : 'var(--tx)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            color: 'var(--sf)', fontSize: 12, fontWeight: 700,
          }}>{initials}</div>
          <button className="btn btn-secondary btn-sm" onClick={onLogout}>Déconnexion</button>
        </div>
      </div>
    </header>
  );
}
