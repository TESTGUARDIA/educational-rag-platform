import { useState, useEffect, useCallback } from 'react';
import { setTokens, clearTokens } from '../utils/api';

export function useAuth() {
  const [me, setMe]       = useState(null);   // { id, nom, role }
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  // Écouter les déconnexions forcées (refresh expiré)
  useEffect(() => {
    const handler = () => { setMe(null); clearTokens(); };
    window.addEventListener('auth:logout', handler);
    return () => window.removeEventListener('auth:logout', handler);
  }, []);

  const login = useCallback(async (email, password) => {
    setLoading(true); setError('');
    try {
      const form = new URLSearchParams({ username: email, password });
      const r = await fetch(
        (window.location.hostname === 'localhost' && window.location.port === '5173'
          ? 'http://localhost:8000' : '/api') + '/auth/login',
        { method: 'POST', body: form }
      );
      if (!r.ok) { const e = await r.json(); throw e; }
      const data = await r.json();
      setTokens(data.access_token, data.refresh_token);
      setMe({ id: data.id, nom: data.nom, role: data.role });
    } catch (e) {
      setError(e.detail || 'Identifiants incorrects');
    } finally {
      setLoading(false);
    }
  }, []);

  const logout = useCallback(() => {
    clearTokens();
    setMe(null);
  }, []);

  return { me, login, logout, loading, error };
}
