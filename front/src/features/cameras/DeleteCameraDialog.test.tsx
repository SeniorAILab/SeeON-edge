import React, { act } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { DeleteCameraDialog } from '@/features/cameras/DeleteCameraDialog';

afterEach(() => {
  document.body.innerHTML = '';
});

describe('DeleteCameraDialog', () => {
  it('uses the labelled shared dialog and initially focuses safe cancel', () => {
    const onCancel = vi.fn();
    const onConfirm = vi.fn();
    const host = document.createElement('div');
    document.body.append(host);
    const root = createRoot(host);
    act(() => root.render(<DeleteCameraDialog camera={{ id: 'cam-1', label: '301호 A', rtsp_url_masked: '', space_id: null, space_name: null, floor_name: null, backend_camera_id: null, status: 'offline', created_at: null }} message={null} onCancel={onCancel} onConfirm={onConfirm} />));
    const dialog = document.querySelector('[role="dialog"]');
    expect(dialog?.textContent).toContain('301호 A 삭제');
    expect(document.activeElement?.textContent).toBe('취소');
    act(() => document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true })));
    expect(onCancel).toHaveBeenCalledTimes(1);
    expect(onConfirm).not.toHaveBeenCalled();
    act(() => root.unmount());
  });

  it('cycles focus and restores the invoker after Escape closes it', () => {
    const host = document.createElement('div');
    document.body.append(host);
    const root = createRoot(host);
    function Harness(): JSX.Element {
      const [open, setOpen] = React.useState(false);
      return <><button onClick={() => setOpen(true)}>삭제 열기</button>{open ? <DeleteCameraDialog camera={{ id: 'cam-1', label: '301호 A', rtsp_url_masked: '', space_id: null, space_name: null, floor_name: null, backend_camera_id: null, status: 'offline', created_at: null }} message={null} onCancel={() => setOpen(false)} onConfirm={vi.fn()} /> : null}</>;
    }
    act(() => root.render(<Harness />));
    const invoker = host.querySelector('button') as HTMLButtonElement;
    act(() => { invoker.focus(); invoker.click(); });
    const buttons = document.querySelectorAll('[role="dialog"] button');
    expect(document.activeElement).toBe(buttons[0]);
    act(() => buttons[0].dispatchEvent(new KeyboardEvent('keydown', { key: 'Tab', shiftKey: true, bubbles: true })));
    expect(document.activeElement).toBe(buttons[buttons.length - 1]);
    act(() => document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true })));
    expect(document.querySelector('[role="dialog"]')).toBeNull();
    expect(document.activeElement).toBe(invoker);
    act(() => root.unmount());
  });
});
