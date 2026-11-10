export default function Modal({ open, onClose, title, subtitle, children, actions }) {
  if (!open) return null;
  return (
    <div className="modal-overlay" onClick={e => e.target === e.currentTarget && onClose()}>
      <div className="modal" role="dialog" aria-modal="true">
        <button className="modal-close" onClick={onClose} aria-label="Fermer">✕</button>
        {title    && <h2 className="modal-title">{title}</h2>}
        {subtitle && <p className="modal-subtitle">{subtitle}</p>}
        {children}
        {actions  && <div className="modal-actions">{actions}</div>}
      </div>
    </div>
  );
}
