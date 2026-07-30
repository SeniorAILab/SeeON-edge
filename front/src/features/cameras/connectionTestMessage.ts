export function connectionFailureMessage(errorClass?: string): string {
  if (errorClass === 'timeout') return '카메라 응답 시간이 초과되었습니다. 네트워크와 전원을 확인하세요.';
  if (errorClass === 'auth') return '카메라 인증에 실패했습니다. 아이디와 비밀번호를 확인하세요.';
  if (errorClass === 'decode') return '영상 형식을 처리할 수 없습니다. 카메라 스트림 설정을 확인하세요.';
  return '카메라 연결을 확인할 수 없습니다. 네트워크와 설정을 확인하세요.';
}
