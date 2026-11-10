// Logo TestGuardAI : le T en chapeau de diplômé (rouge correcteur),
// le G en loupe (violet IA), le nom en Bricolage Grotesque ExtraBold.
const RED = '#D83A2E';
const VIOLET = '#6D4AE0';
const VIOLET_TINT = '#ECE6FC';
const IVORY = '#FFFDF8';

function CapT() {
  return (
    <svg className="t" viewBox="0 0 80 72" aria-hidden="true">
      <path d="M40 13C52 13 64 14 70 16V33" stroke={RED} strokeWidth="3.4" fill="none" strokeLinecap="round" strokeLinejoin="round" />
      <rect x="66.4" y="31" width="7.2" height="15" rx="3.6" fill={RED} />
      <rect x="34" y="16" width="12" height="55" rx="5" fill="currentColor" />
      <polygon points="40,1 78,13 40,25 2,13" fill="currentColor" />
      <path d="M40 7 64 14.5" stroke={IVORY} strokeWidth="1.8" strokeLinecap="round" />
      <circle cx="40" cy="13" r="3.2" fill={RED} />
    </svg>
  );
}

function LoupeG({ dark }) {
  const ring = dark ? '#A991FF' : VIOLET;
  return (
    <svg className="g" viewBox="0 0 72 72" aria-hidden="true">
      <circle cx="34" cy="36" r="19" fill={dark ? '#2B2A33' : VIOLET_TINT} />
      <path d="M24 28A13 13 0 0 1 33 22.5" stroke={RED} strokeWidth="3.5" fill="none" strokeLinecap="round" />
      <path d="M53.9 19.3A26 26 0 1 0 53.9 52.7" stroke={ring} strokeWidth="11" fill="none" strokeLinecap="round" />
      <path d="M53.9 52.7 66 66" stroke={ring} strokeWidth="12" strokeLinecap="round" />
      <path d="M37 36H58" stroke={ring} strokeWidth="10" strokeLinecap="round" />
    </svg>
  );
}

/**
 * variant : "inline" (en-tête) ou "stacked" (page de connexion, grands formats)
 * size    : taille de police en px
 * dark    : version sur fond sombre
 */
export default function Logo({ variant = 'inline', size = 24, dark = false }) {
  const cls = `tgl${variant === 'stacked' ? ' st' : ''}${dark ? ' dark' : ''}`;
  if (variant === 'stacked') {
    return (
      <div className={cls} style={{ fontSize: size }} role="img" aria-label="TestGuardAI">
        <div><CapT />est</div>
        <div><LoupeG dark={dark} />uard<span className="tgl-ai">AI</span></div>
      </div>
    );
  }
  return (
    <span className={cls} style={{ fontSize: size }} role="img" aria-label="TestGuardAI">
      <CapT />est<LoupeG dark={dark} />uard<span className="tgl-ai">AI</span>
    </span>
  );
}
