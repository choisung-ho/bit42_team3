from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import base64
import json
import asyncio
import uvicorn
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager # Lifespan 관리를 위해 추가

# --- 실제 Gemini API 연동을 위한 임포트 ---
from google import genai
from google.genai import types

# TODO: 실제 STT (Speech-to-Text)를 사용하려면 'google-cloud-speech' SDK와 인증 설정이 필요합니다.

# -----------------
# FastAPI Lifespan Context Manager 정의
# -----------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    서버 시작 시 Gemini 클라이언트를 초기화하고, 종료 시 정리합니다.
    """
    global client
    try:
        # 서버 시작 시: Gemini 클라이언트 초기화 (동기 방식)
        app.state.client = genai.Client()
        print("Gemini Client initialized and attached to app state.")
    except Exception as e:
        print(f"Gemini Client Initialization Error: {e}. Running in Mock mode.")
        app.state.client = None
    
    yield # 서버 실행 시작
    
    # 서버 종료 시: 리소스 정리 (현재 genai 클라이언트는 별도 정리 로직 불필요)
    print("FastAPI application shutting down.")


# lifespan 함수를 FastAPI 앱에 연결
app = FastAPI(title="감성 기록 AI 분석 서버", lifespan=lifespan)

# -----------------
# 1. 요청/응답 데이터 모델 정의 (Pydantic)
# -----------------
class CallAnalysisRequest(BaseModel):
    """Flutter 앱으로부터 받는 요청 데이터 모델"""
    user_id: str
    audio_base64: str  # 녹음 파일 (Base64 인코딩)
    call_metadata: dict = Field(description="통화 상대, 시간, GPS 등 메타데이터")

class AnalysisResult(BaseModel):
    """분석 결과 응답 모델"""
    success: bool
    diary_draft: dict  # LLM이 생성한 구조화된 일기 초안 섹션
    emotion_score: int
    calendar_event: str | None # 추출된 일정

# -----------------
# 1.5. LLM 결과의 Pydantic 스키마 정의 (Gemini에게 요구할 JSON 형식)
# -----------------
class CallSummary(BaseModel):
    """통화 요약 상세 모델"""
    keywords: List[str] = Field(description="통화 내용에서 추출된 3-5개의 핵심 키워드 목록.")
    dialogue: List[str] = Field(description="화자 구분된 텍스트를 대화 형식으로 나눈 목록.")

class DiarySection(BaseModel):
    """일기 초안의 핵심 섹션"""
    time: str = Field(description="분석된 통화/이벤트 발생 시각 (예: '14:30').")
    type: str = Field(description="이벤트 유형 (예: 'call_analysis').")
    summary: str = Field(description="이 이벤트에 대한 감성적인 요약 설명.")
    call_summary: CallSummary
    
class LLMStructuredOutput(BaseModel):
    """Gemini가 최종적으로 반환할 JSON 구조"""
    diary_sections: DiarySection = Field(description="통화 내용을 기반으로 생성된 일기 초안 섹션.")
    emotion_score: int = Field(description="통화 내용의 전반적인 긍정/부정 감성을 0부터 100 사이의 점수로 표현.")
    calendar_event: Optional[str] = Field(description="통화 내용에서 추출된 명확한 일정(없으면 null).")

# -----------------
# 2. STT 및 LLM 로직 구현
# -----------------

# Mock STT 결과 (실제 오디오가 없으므로 임시로 사용할 텍스트)
MOCK_STT_TRANSCRIPT = (
    "화자 A: 안녕하세요, 우송대 AI 빅데이터학과 정한결입니다.\n"
    "화자 B: 네, 저는 김교수입니다. 프로젝트 주제는 잘 잡았나요?\n"
    "화자 A: 네, '감성 기록' 앱으로 잡았습니다. LLM으로 통화 내용을 분석해요.\n"
    "화자 B: 아주 흥미롭네요. 다음 주 화요일 오후 3시에 중간 보고서 제출해 주세요."
)


async def run_stt_with_diarization(audio_bytes: bytes) -> str:
    """
    [Mock 함수] STT API를 호출하여 오디오를 텍스트로 변환하고 화자를 구분합니다.
    실제 구현 시: Google Cloud Speech-to-Text SDK 사용
    """
    # STT API 호출을 시뮬레이션하기 위해 3초 대기합니다.
    await asyncio.sleep(3) 
    print(f"Debug: 수신된 오디오 바이트 크기: {len(audio_bytes)}")
    
    return MOCK_STT_TRANSCRIPT

async def generate_diary_draft_with_gemini(stt_result: str, metadata: dict) -> dict:
    """
    [실제 구현] Gemini LLM API를 호출하여 STT 결과 기반으로 일기 초안 JSON을 생성합니다.
    """
    # Lifespan에서 초기화된 클라이언트를 app.state에서 가져옵니다.
    client = app.state.client
    
    # 클라이언트가 초기화되지 않았거나 API 키가 설정되지 않은 경우 Mock 데이터 반환
    if client is None:
        await asyncio.sleep(1)
        # Mock 데이터는 LLMStructuredOutput 스키마를 따릅니다.
        return {
            "diary_sections": {
                "time": "14:30",
                "type": "call_analysis",
                "summary": "김교수님과 프로젝트 중간 점검 통화가 있었습니다. 주제가 좋다는 긍정적인 평가를 받았습니다.",
                "call_summary": {
                    "keywords": ["프로젝트", "중간 보고서", "긍정적 피드백"],
                    "dialogue": stt_result.split('\n'),
                }
            },
            "emotion_score": 85,
            "calendar_event": "프로젝트 중간 보고서 제출 (다음 주 화요일 오후 3시)"
        }

    print("--- Gemini API 호출 시작 ---")
    
    # LLM이 따라야 할 프롬프트 및 시스템 지침
    system_instruction = (
        "당신은 사용자의 통화 내용을 분석하여 일기를 자동으로 작성하는 AI 비서입니다. "
        "다음 STT 텍스트와 메타데이터를 분석하여 사용자의 감정 상태를 파악하고, "
        "캘린더에 등록할 일정이 있다면 추출하며, 응답은 반드시 제공된 JSON 스키마를 따라야 합니다."
        f"통화 메타데이터: {json.dumps(metadata, ensure_ascii=False)}"
    )

    # Gemini API 호출 (동기적 호출이므로 비동기 환경에 맞게 처리)
    loop = asyncio.get_event_loop()
    
    # contents에 STT 결과 텍스트를 담아 LLM에 전송
    response = await loop.run_in_executor(
        None, # 기본 스레드 풀 사용
        lambda: client.models.generate_content(
            model='gemini-2.5-flash',
            contents=stt_result,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                response_schema=LLMStructuredOutput, # 정의한 Pydantic 스키마 사용
            )
        )
    )

    # Gemini가 반환한 JSON 텍스트를 파이썬 딕셔너리로 변환
    try:
        json_text = response.text.strip()
        # LLM이 종종 ```json ... ``` 형태로 반환할 수 있으므로 클리닝
        if json_text.startswith("```json"):
            json_text = json_text.strip("```json").strip("```").strip()
            
        llm_output_dict = json.loads(json_text)
        return llm_output_dict
    except json.JSONDecodeError as e:
        print(f"LLM 응답 JSON 디코딩 실패: {e}")
        print(f"RAW 응답: {response.text}")
        # 실패 시 예외 처리
        raise HTTPException(status_code=500, detail="LLM returned invalid JSON structure.")
    except Exception as e:
        print(f"Gemini 호출 중 알 수 없는 오류 발생: {e}")
        raise HTTPException(status_code=500, detail="Gemini API Call Failed")


async def process_stt_and_llm(audio_bytes: bytes, metadata: dict) -> dict:
    """오디오 바이트를 받아 STT 및 LLM 분석을 수행하는 비동기 함수"""
    try:
        print("--- 1. STT 및 화자 구분 시작 (Mock) ---")
        # STT Mock 함수 호출
        stt_result = await run_stt_with_diarization(audio_bytes)
        print("--- 2. STT 완료, LLM 분석 시작 ---")
        
        # LLM 분석 함수 호출 (실제 API 호출 또는 Mock)
        llm_output = await generate_diary_draft_with_gemini(stt_result, metadata)
        print("--- 3. LLM 분석 완료 ---")
        
        # 4. DB 저장 로직 (나중에 구현)
        # await save_analysis_to_db(metadata['user_id'], llm_output)

        return llm_output # LLM이 생성한 JSON 데이터를 반환
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"AI 처리 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail="AI Analysis Failed")


# -----------------
# 3. FastAPI 엔드포인트 정의
# -----------------

@app.post("/analyze/call", response_model=AnalysisResult)
async def analyze_call_record(request: CallAnalysisRequest):
    """
    통화 녹음 파일(Base64)을 받아 STT 및 LLM 분석을 수행하고 결과를 반환합니다.
    """
    if not request.audio_base64:
         raise HTTPException(status_code=400, detail="audio_base64 field is empty")

    try:
        # Base64 디코딩 (파일 데이터를 바이트 형태로 변환)
        audio_bytes = base64.b64decode(request.audio_base64)
        
        if len(audio_bytes) == 0:
            # 이 오류는 Base64 디코딩 후 데이터가 없다는 의미입니다.
            raise HTTPException(status_code=400, detail="Decoded audio data size is zero.")

        # AI 분석 파이프라인 실행
        llm_result_json = await process_stt_and_llm(audio_bytes, request.call_metadata)
        
        # LLM 결과를 응답 모델에 맞춰 반환 (Pydantic 유효성 검사)
        return AnalysisResult(
            success=True,
            diary_draft=llm_result_json.get("diary_sections", {}), 
            emotion_score=llm_result_json.get("emotion_score", 50),
            calendar_event=llm_result_json.get("calendar_event")
        )
        
    except HTTPException:
        # 이미 내부 함수에서 발생시킨 예외를 그대로 전파
        raise
    except Exception as e:
        # 기타 모든 예외 처리
        print(f"API 핸들러에서 알 수 없는 오류 발생: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid Request or Unknown Error: {e}")
        
if __name__ == "__main__":
    # 서버 실행 명령어 (자동 재시작 기능 포함)
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)