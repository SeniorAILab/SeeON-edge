import { useCallback, useEffect, useState } from 'react';

const STORAGE_KEY = 'ml-dashboard-theme';

function prefersDark(): boolean {
  if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') {
    return false;
  }
  return window.matchMedia('(prefers-color-scheme: dark)').matches;
}

function initialDark(): boolean {
  if (typeof window === 'undefined') {
    return false;
  }
  const stored = window.localStorage.getItem(STORAGE_KEY);
  if (stored === 'dark') return true;
  if (stored === 'light') return false;
  return prefersDark();
}

function applyDarkClass(dark: boolean): void {
  if (typeof document === 'undefined') return;
  document.documentElement.classList.toggle('dark', dark);
}

/**
 * Owns the `.dark` class on <html> and persists the operator's choice.
 * Mirrors front's token-based theme so ml-dashboard follows the same palette.
 */
export function useDarkMode(): { dark: boolean; toggle: () => void } {
  const [dark, setDark] = useState<boolean>(initialDark);

  useEffect(() => {
    applyDarkClass(dark);
    if (typeof window !== 'undefined') {
      window.localStorage.setItem(STORAGE_KEY, dark ? 'dark' : 'light');
    }
  }, [dark]);

  const toggle = useCallback(() => setDark((current) => !current), []);

  return { dark, toggle };
}
