// ── Toast global (singleton) ─────────────────────────────────────────────────
let _setToast = null;

export function registerToast(fn) { _setToast = fn; }

export function toast(msg) {
  if (_setToast) _setToast(msg);
}
