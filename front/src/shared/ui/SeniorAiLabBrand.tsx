import { useId } from 'react';

export function SeniorAiLabBrand({ inverse = false, compact = false }: { inverse?: boolean; compact?: boolean }): JSX.Element {
  const gradientId = `sa-grad-${useId().replace(/:/g, '')}`;
  return (
    <div className={`sa-brand${compact ? ' sa-brand-compact' : ''}`} aria-label="Senior AI Lab ML Control">
      <svg viewBox="0 0 100 100" fill="none" aria-hidden="true">
        <defs>
          <linearGradient id={gradientId} x1="20" y1="14" x2="78" y2="84" gradientUnits="userSpaceOnUse">
            <stop stopColor="#2BB6A3" />
            <stop offset="0.5" stopColor="#2F6FB0" />
            <stop offset="1" stopColor="#16325A" />
          </linearGradient>
        </defs>
        <path d="M67 25 C59 18 43 18 36 25 C27 34 32 45 46 49 C60 53 65 61 59 69 C52 78 37 78 29 71" stroke={`url(#${gradientId})`} strokeWidth="11" strokeLinecap="round" />
        <path d="M64 84 L79 33 L94 84 M69 70 H89" stroke="#16325A" strokeWidth="9.5" strokeLinecap="round" strokeLinejoin="round" />
        <g stroke="#2F6FB0" strokeWidth="2.4" strokeLinecap="round"><path d="M55 47 L62 40 L62 28" /><path d="M62 40 L72 34 L72 24" /><path d="M62 40 L74 44 L82 40" /><path d="M55 47 L52 38" /></g>
        <g fill="#2BB6A3"><circle cx="62" cy="26" r="3.4" /><circle cx="72" cy="22" r="3.4" /></g>
        <g fill="#2F6FB0"><circle cx="52" cy="36" r="3.4" /><circle cx="83" cy="40" r="3.4" /></g>
      </svg>
      <span><strong className={inverse ? 'inverse' : ''}>Senior AI Lab</strong>{compact ? null : <small>ML Control</small>}</span>
    </div>
  );
}
