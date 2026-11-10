// ── API client avec gestion JWT + refresh token automatique ─────────────────
const API = window.location.hostname === 'localhost' && window.location.port === '5173'
  ? 'http://localhost:8000'
  : '/api';

let accessToken = null;
let refreshToken = null;

export function setTokens(access, refresh) {
  accessToken = access;
  refreshToken = refresh;
}

export function clearTokens() {
  accessToken = null;
  refreshToken = null;
}

export function hasToken() {
  return !!accessToken;
}

export async function apiFetch(method, path, body, isRetry = false) {
  const isForm = body instanceof FormData;
  // FormData : le navigateur fixe lui-même le Content-Type multipart (avec la frontière)
  const opts = { method, headers: isForm ? {} : { 'Content-Type': 'application/json' } };
  if (accessToken) opts.headers['Authorization'] = 'Bearer ' + accessToken;
  if (body) opts.body = isForm ? body : JSON.stringify(body);

  const r = await fetch(API + path, opts);

  // 401 → tentative de refresh automatique
  if (r.status === 401 && !isRetry && refreshToken) {
    const ref = await fetch(API + '/auth/refresh', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ refresh_token: refreshToken }),
    });
    if (ref.ok) {
      const data = await ref.json();
      setTokens(data.access_token, data.refresh_token);
      return apiFetch(method, path, body, true);
    } else {
      clearTokens();
      window.dispatchEvent(new Event('auth:logout'));
      return;
    }
  }

  if (!r.ok) throw await r.json();
  return r.json();
}

export async function apiUpload(path, formData) {
  const opts = { method: 'POST', body: formData };
  if (accessToken) opts.headers = { Authorization: 'Bearer ' + accessToken };
  const r = await fetch(API + path, opts);
  if (r.status === 401) { clearTokens(); window.dispatchEvent(new Event('auth:logout')); return; }
  if (!r.ok) throw await r.json();
  return r.json();
}
