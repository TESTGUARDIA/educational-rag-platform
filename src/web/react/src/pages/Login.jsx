import { useState } from 'react';
import Logo from '../components/Logo';

export default function Login({ onLogin, loading, error }) {
  const [email, setEmail] = useState('');
  const [pass, setPass]   = useState('');
  const [role, setRole]   = useState('teacher');

  function handleSubmit(e) {
    e.preventDefault();
    onLogin(email, pass);
  }

  return (
    <div style={{
      minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center',
      background: 'var(--bg)', padding: 20,
    }}>
      <div style={{
        width: '100%', maxWidth: 440, background: 'var(--sf)',
        border: '1px solid var(--bd)', borderRadius: 10, padding: '44px 40px',
      }}>
        <div style={{ marginBottom: 30 }}><Logo variant="stacked" size={56} /></div>

        <h1 style={{ fontSize: 24, marginBottom: 6 }}>Bienvenue</h1>
        <p style={{ color: 'var(--mu)', fontSize: 14, marginBottom: 24 }}>
          Choisissez votre espace pour vous connecter.
        </p>

        <div role="radiogroup" aria-label="Espace" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 24 }}>
          {[
            { id: 'teacher', label: 'Enseignant', hint: 'Cours, copies, corrections' },
            { id: 'student', label: 'Élève', hint: 'Évaluations et résultats' },
          ].map(r => {
            const active = role === r.id;
            return (
              <button
                key={r.id} type="button" role="radio" aria-checked={active}
                onClick={() => setRole(r.id)}
                style={{
                  textAlign: 'left', padding: '14px 14px', borderRadius: 8, cursor: 'pointer',
                  fontFamily: 'inherit', background: active ? 'var(--grl)' : 'transparent',
                  border: active ? '1.5px solid var(--gr)' : '1.5px solid var(--bd)', color: 'var(--tx)',
                }}
              >
                <span style={{ display: 'block', fontWeight: 700, fontSize: 15, marginBottom: 3 }}>{r.label}</span>
                <span style={{ display: 'block', fontSize: 12, color: 'var(--mu)' }}>{r.hint}</span>
              </button>
            );
          })}
        </div>

        <form onSubmit={handleSubmit}>
          <div className="fg">
            <label htmlFor="linput">Adresse e-mail</label>
            <input id="linput" type="email" value={email} onChange={e => setEmail(e.target.value)}
              placeholder="votre@email.fr" autoComplete="email" required />
          </div>
          <div className="fg">
            <label htmlFor="pinput">Mot de passe</label>
            <input id="pinput" type="password" value={pass} onChange={e => setPass(e.target.value)}
              autoComplete="current-password" required />
          </div>
          {error && (
            <p role="alert" style={{ color: 'var(--tx)', fontWeight: 600, fontSize: 13, marginBottom: 12 }}>{error}</p>
          )}
          <button type="submit" className="btn btn-primary btn-full" disabled={loading}>
            {loading ? 'Connexion…' : 'Se connecter'}
          </button>
        </form>

        <p style={{ textAlign: 'center', fontSize: 12, color: 'var(--mu)', marginTop: 16 }}>
          Démo : marie.dupont@testguard.ai, mot de passe « password »
        </p>
      </div>
    </div>
  );
}
