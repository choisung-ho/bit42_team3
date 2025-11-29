# GitHub 사용 방법 가이드

## 목차
1. [Git 기본 설정](#git-기본-설정)
2. [저장소 클론하기](#저장소-클론하기)
3. [브랜치 관리](#브랜치-관리)
4. [변경사항 커밋하기](#변경사항-커밋하기)
5. [Pull Request 생성하기](#pull-request-생성하기)

---

## Git 기본 설정

Git을 처음 사용할 때 사용자 정보를 설정해야 합니다.

```bash
# 사용자 이름 설정
git config --global user.name "홍길동"

# 이메일 설정
git config --global user.email "your-email@example.com"
```

---

## 저장소 클론하기

GitHub에서 프로젝트를 로컬로 복제하려면 다음 명령어를 사용합니다.

```bash
# HTTPS 방식
git clone https://github.com/choisung-ho/bit42_team3.git

# 클론한 디렉토리로 이동
cd bit42_team3
```

---

## 브랜치 관리

### 새 브랜치 생성 및 이동

```bash
# 새 브랜치 생성 및 이동
git checkout -b feature/새기능이름

# 브랜치 목록 확인
git branch

# 특정 브랜치로 이동
git checkout main
```

### 브랜치 삭제

```bash
# 로컬 브랜치 삭제
git branch -d feature/삭제할브랜치

# 원격 브랜치 삭제
git push origin --delete feature/삭제할브랜치
```

---

## 변경사항 커밋하기

### 기본 워크플로우

```bash
# 현재 상태 확인
git status

# 변경된 파일 스테이징
git add 파일이름
# 또는 모든 변경사항 스테이징
git add .

# 커밋 메시지와 함께 커밋
git commit -m "커밋 메시지: 변경 내용 설명"

# 원격 저장소에 푸시
git push origin 브랜치이름
```

### 커밋 메시지 컨벤션 예시

- `feat:` 새로운 기능 추가
- `fix:` 버그 수정
- `docs:` 문서 수정
- `style:` 코드 포맷팅
- `refactor:` 코드 리팩토링
- `test:` 테스트 코드 추가

```bash
git commit -m "feat: 로그인 기능 추가"
git commit -m "fix: 회원가입 버그 수정"
git commit -m "docs: README 업데이트"
```

---

## Pull Request 생성하기

1. 작업한 브랜치를 원격 저장소에 푸시합니다.
   ```bash
   git push origin feature/새기능이름
   ```

2. GitHub 웹사이트에서 해당 저장소로 이동합니다.

3. "Compare & pull request" 버튼을 클릭합니다.

4. PR 제목과 설명을 작성합니다:
   - **제목**: 변경 내용을 간결하게 요약
   - **설명**: 변경 이유, 주요 변경사항, 테스트 방법 등

5. "Create pull request" 버튼을 클릭합니다.

6. 팀원의 코드 리뷰를 받고, 승인 후 병합합니다.

---

## 유용한 Git 명령어

```bash
# 원격 저장소 변경사항 가져오기
git pull origin main

# 변경 이력 확인
git log --oneline

# 마지막 커밋 수정
git commit --amend

# 변경사항 임시 저장
git stash
git stash pop
```

---

## 문제 해결

### 충돌(Conflict) 해결

1. 충돌이 발생한 파일을 열어 충돌 부분을 확인합니다.
2. `<<<<<<< HEAD`와 `>>>>>>>` 사이의 내용을 정리합니다.
3. 수정 후 다시 커밋합니다.

```bash
git add .
git commit -m "fix: merge conflict 해결"
```

---

더 자세한 정보는 [GitHub 공식 문서](https://docs.github.com/ko)를 참고하세요.
