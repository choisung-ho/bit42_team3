import requests
import base64
import json

# 서버 주소 및 엔드포인트 설정
SERVER_URL = "http://127.0.0.1:8000/analyze/call"
# (참고: 서버가 Docker나 다른 환경에서 실행 중인 경우, IP 주소를 변경해야 합니다.)

def create_mock_request_data():
    """
    FastAPI 서버의 CallAnalysisRequest 모델에 맞는 모의(Mock) 데이터를 생성합니다.
    
    실제 Flutter 앱에서는 통화 녹음 파일을 읽어와 Base64로 인코딩해야 합니다.
    """
    # 1. Base64 인코딩된 '가짜' 오디오 데이터 생성
    # - 실제 오디오 파일이 없으므로, 아주 작은 임시 데이터만 인코딩합니다.
    mock_audio_data = b"This is mock audio data for the call recording."
    encoded_audio = base64.b64encode(mock_audio_data).decode('utf-8')
    
    # 2. 메타데이터 (Flutter 앱에서 보내줄 정보)
    metadata = {
        "with_person": "김교수",
        "duration_sec": 300,
        "location": "우송대학교 강의실 A",
        "timestamp": "2025-12-03T14:20:00Z"
    }

    # 3. 최종 요청 데이터 구성
    request_data = {
        "user_id": "han-gyeol-25",
        "audio_base64": encoded_audio, # 서버가 요구하는 핵심 데이터
        "call_metadata": metadata
    }
    
    return request_data

def get_analysis_result():
    """FastAPI 서버에 요청을 보내고 결과를 수신합니다."""
    # 요청에 사용할 데이터 준비
    data = create_mock_request_data()
    
    print(f"요청 전송 시작: {SERVER_URL}")
    print("서버의 STT/LLM 처리 시간을 기다리는 중입니다 (Mock: 약 8초 소요)...")

    response = None
    try:
        # HTTP POST 요청 전송
        response = requests.post(
            SERVER_URL, 
            json=data, 
            headers={"Content-Type": "application/json"},
            timeout=30 # 서버 응답 대기 시간 설정
        )
        
        # 200번대 상태 코드가 아니면 예외 발생 (예: 400, 500 에러)
        response.raise_for_status() 
        
        # 최종 결과 값 (JSON) 반환
        result = response.json()
        print("\n=== 서버 응답 수신 성공 ===")
        # 결과를 보기 좋게(들여쓰기 4칸, 한글 깨짐 방지) 출력
        print(json.dumps(result, indent=4, ensure_ascii=False)) 
        
        return result

    except requests.exceptions.RequestException as e:
        print(f"\n❌ 요청 실패 또는 서버 오류 발생: {e}")
        if response is not None:
             print(f"상태 코드: {response.status_code}")
             print(f"응답 본문: {response.text}")
        return None

if __name__ == "__main__":
    get_analysis_result()