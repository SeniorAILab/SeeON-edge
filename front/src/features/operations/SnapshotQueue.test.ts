import { afterEach, describe, expect, it, vi } from 'vitest';
import { SnapshotQueue } from '@/features/operations/SnapshotQueue';

afterEach(() => vi.useRealTimers());

describe('SnapshotQueue', () => {
  it('limits unresolved requests to six and drains the queue', () => {
    const queue = new SnapshotQueue((id, key) => `/snapshot/${id}?refresh=${key}`);
    queue.update(Array.from({ length: 13 }, (_, index) => ({ id: `cam-${index}`, mediaId: `cam-${index}` })), true);
    expect(queue.snapshot().filter((entry) => entry.requestUrl).length).toBe(6);
    expect(queue.unresolvedCount).toBe(6);

    queue.resolve('cam-0', 'cam-0', 'loaded');
    expect(queue.unresolvedCount).toBe(6);
    expect(queue.get('cam-6')?.requestUrl).toBe('/snapshot/cam-6?refresh=0');
    expect(queue.get('cam-0')?.state).toBe('loaded');
  });

  it('refreshes with deterministic delay, exposes stale, and retries errors', () => {
    vi.useFakeTimers();
    const queue = new SnapshotQueue((id, key) => `/snapshot/${id}?refresh=${key}`);
    queue.update([{ id: 'ok', mediaId: 'ok' }, { id: 'bad', mediaId: 'bad' }], true);
    queue.resolve('ok', 'ok', 'loaded');
    queue.resolve('bad', 'bad', 'error');
    expect(queue.get('bad')?.state).toBe('error');

    vi.advanceTimersByTime(6_000);
    expect(queue.get('ok')?.state).toBe('stale');
    expect(queue.get('ok')?.requestUrl).toContain('refresh=1');
    expect(queue.get('bad')?.requestUrl).toContain('refresh=1');
  });

  it('cancels removed work and pauses every request off-page', () => {
    vi.useFakeTimers();
    const queue = new SnapshotQueue((id, key) => `/snapshot/${id}?refresh=${key}`);
    queue.update(['a', 'b', 'c'].map((id) => ({ id, mediaId: id })), true);
    queue.resolve('a', 'a', 'loaded');
    queue.update([{ id: 'b', mediaId: 'b' }], true);
    expect(queue.get('a')).toBeUndefined();
    expect(queue.get('c')).toBeUndefined();
    queue.update([{ id: 'b', mediaId: 'b' }], false);
    expect(queue.unresolvedCount).toBe(0);
    expect(queue.get('b')?.requestUrl).toBeNull();
    vi.advanceTimersByTime(10_000);
    expect(queue.get('b')?.requestUrl).toBeNull();
  });

  it('times out an unresolved snapshot and releases its slot', () => {
    vi.useFakeTimers();
    const queue = new SnapshotQueue((id, key) => `/snapshot/${id}?refresh=${key}`);
    queue.update(Array.from({ length: 7 }, (_, index) => ({ id: `cam-${index}`, mediaId: `cam-${index}` })), true);
    expect(queue.get('cam-6')?.requestUrl).toBeNull();
    vi.advanceTimersByTime(10_000);
    expect(queue.get('cam-0')?.state).toBe('error');
    expect(queue.get('cam-6')?.requestUrl).not.toBeNull();
    expect(queue.unresolvedCount).toBeLessThanOrEqual(6);
  });

  it('invalidates in-flight work and last-good data when the media identity changes', () => {
    const queue = new SnapshotQueue((mediaId, key) => `/snapshot/${mediaId}?refresh=${key}`);
    queue.update([{ id: 'local', mediaId: 'worker-old' }], true);
    const oldRequest = queue.get('local')?.requestUrl;
    queue.resolve('local', 'worker-old', 'loaded', oldRequest ?? undefined);
    expect(queue.get('local')?.lastLoadedUrl).toBe(oldRequest);

    queue.update([{ id: 'local', mediaId: 'worker-new' }], true);
    const newRequest = queue.get('local')?.requestUrl;
    expect(queue.get('local')?.mediaId).toBe('worker-new');
    expect(queue.get('local')?.lastLoadedUrl).toBeNull();

    queue.resolve('local', 'worker-new', 'error', newRequest ?? undefined);
    queue.resolve('local', 'worker-old', 'loaded', oldRequest ?? undefined);

    expect(queue.get('local')?.state).toBe('error');
    expect(queue.get('local')?.lastLoadedUrl).toBeNull();
  });
});
