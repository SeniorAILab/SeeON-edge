import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { describe, expect, it } from 'vitest';

function readCssWithLocalImports(path: string): string {
  const source = readFileSync(path, 'utf8');
  return source.replace(/@import\s+["'](\.[^"']+)["']\s*;/g, (_statement, importPath: string) => (
    readCssWithLocalImports(resolve(dirname(path), importPath))
  ));
}

describe('EventHistoryPage filter touch targets', () => {
  it('locks the event filter controls to the shared 44px minimum', () => {
    const page = readFileSync('src/features/events/EventHistoryPage.tsx', 'utf8');
    expect(page.match(/className="event-filter-control"/g)).toHaveLength(4);
    expect(readCssWithLocalImports('src/styles.css')).toMatch(/\.event-filter-control\s*\{[^}]*min-height:\s*44px;/s);
  });
});
