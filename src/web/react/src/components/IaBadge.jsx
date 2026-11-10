// Badge de détection IA : niveau calé sur le seuil appris par le modèle, pas un seuil fixe.
// Faible (gris) · à vérifier (contour noir, de 50 % au seuil) · alerte (noir plein, au-dessus du seuil).
export default function IaBadge({ ia, prefix = '' }) {
  if (!ia) return <span style={{ color:'var(--mu)' }}>—</span>;

  const st = ia.statut_analyse;
  if (st === 'en_cours') return <span className="badge badge-gray">analyse…</span>;
  if (st === 'non_analysé' || st === 'non_analyse')
    return <span className="badge badge-gray" title={ia.message || ''}>non analysé</span>;
  if (st === 'erreur') return <span className="badge badge-gray" title={ia.message || ''}>erreur</span>;
  if (ia.score_ia_global == null) return <span style={{ color:'var(--mu)' }}>—</span>;

  const score = ia.score_ia_global;
  const seuil = ia.seuil != null ? ia.seuil : 0.5;
  const [cls, niveau] =
    score >= seuil              ? ['badge-alert', 'Alerte']
    : score >= Math.min(0.5, seuil) ? ['badge-check', 'À vérifier']
    : ['badge-low', 'Faible'];
  const title = `Seuil du modèle : ${Math.round(seuil * 100)} %${ia.message ? ` — ${ia.message}` : ''}`;

  return (
    <span className={`badge ${cls}`} title={title}>
      {prefix}{niveau} · {Math.round(score * 100)} %
    </span>
  );
}
