// Palette des matières de la charte : couleurs vives choisies par l'enseignant.
// Les rouges et violets en sont exclus pour ne jamais se confondre avec la marque.
// "texte" = couleur des initiales sur la tuile ; "teinte"/"etiquette" = étiquette lisible.
export const MATIERES_PALETTE = [
  { nom: 'Mandarine', couleur: '#F28C28', texte: '#1C1B22', teinte: '#FDEBD7', etiquette: '#945A25' },
  { nom: 'Soleil',    couleur: '#F5C02E', texte: '#1C1B22', teinte: '#FDF3D8', etiquette: '#846A28' },
  { nom: 'Pomme',     couleur: '#8CC63F', texte: '#1C1B22', teinte: '#EDF4DA', etiquette: '#567431' },
  { nom: 'Émeraude',  couleur: '#1FA463', texte: '#FFFDF8', teinte: '#DBEFE0', etiquette: '#1E754D' },
  { nom: 'Sarcelle',  couleur: '#0F766E', texte: '#FFFDF8', teinte: '#D9E7E2', etiquette: '#106F68' },
  { nom: 'Lagon',     couleur: '#12A4B6', texte: '#1C1B22', teinte: '#D9EFED', etiquette: '#167381' },
  { nom: 'Océan',     couleur: '#2A6FDB', texte: '#FFFDF8', teinte: '#DDE6F3', etiquette: '#2862BD' },
  { nom: 'Corail',    couleur: '#FF7A5C', texte: '#1C1B22', teinte: '#FFE8DF', etiquette: '#A05244' },
  { nom: 'Terre',     couleur: '#9A5B34', texte: '#FFFDF8', teinte: '#EFE3D9', etiquette: '#905633' },
  { nom: 'Ardoise',   couleur: '#4F6375', texte: '#FFFDF8', teinte: '#E3E4E3', etiquette: '#4F6375' },
];

export function styleMatiere(couleur) {
  return MATIERES_PALETTE.find(p => p.couleur === couleur) || MATIERES_PALETTE[6];
}

export function initialesMatiere(nom = '') {
  const n = nom.trim();
  return n ? n.charAt(0).toUpperCase() + n.slice(1, 2).toLowerCase() : '?';
}
