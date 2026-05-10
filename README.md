# 메이플 운빨 리포트

Nexon Open API 기반으로 메이플스토리 큐브, 잠재능력 재설정, 스타포스 과거 기록을 불러와 개인 관측 기록을 정리하는 Streamlit 대시보드입니다.

이 프로젝트는 특정 시간이나 조건을 단정적으로 권유하는 방식이 아니라, 과거 기록상 어떤 조건에서 결과가 상대적으로 높게 또는 아쉽게 관측되었는지 참고용으로 보여주는 분석 서비스입니다.

## 프로젝트 소개

- 메이플 유저가 가장 궁금해하는 질문인 “언제 결과가 좋게 관측됐나?”를 과거 기록 기준으로 정리합니다.
- 큐브/잠재능력 재설정은 주요옵션 출현률, 유효옵션 출현률, 등급업률을 나눠 봅니다.
- 스타포스는 성공률과 파괴율을 함께 보며, 날짜/요일/시간대/구간 단위로 비교합니다.
- 모든 해석은 과거 기록 기반 참고용이며, 향후 결과를 보장하지 않습니다.

## 주요 기능

- Nexon Open API Key 직접 입력 기반 데이터 조회
- 큐브, 잠재능력 재설정, 스타포스 전체 데이터 수집
- 최근 2년 기준 조회 가능 기간 자동 계산 및 날짜 범위 자동 보정
- 큐브/잠재능력:
  - 주요옵션 출현률
  - 유효옵션 출현률
  - 등급업률
  - 큐브 타입별 / 요일별 / 시간대별 / 날짜별 비교
- 스타포스:
  - 성공률
  - 파괴율
  - 스타포스 구간별 / 전이별 / 요일별 / 시간대별 / 날짜별 비교
- 과거 기록상 좋게 관측된 조건 TOP 5 / 아쉬웠던 조건 TOP 5
- 날짜별로 좋게 관측된 날 / 아쉬웠던 날 TOP 5
- 화면 안에서만 분석 결과 표시
- API 디버그 탭 제공

## 화면 구조

- `종합 요약`
- `일자별 분석`
- `시간별 분석`
- `요일별 분석`
- `조건 조합 TOP 5`
- `원본 데이터`
- `API 디버그`

상단에는 메이플 유저에게 익숙한 프로필형 헤더를 두고, 닉네임, 월드, 직업, 최근 동기화 시각, 조회 기간을 먼저 보여줍니다.

## 주요옵션 / 유효옵션 정의

큐브 옵션 분석은 아래 기준을 따릅니다.

- 주요옵션 출현률:
  - 사용자가 선택한 주요 옵션이 1줄 이상 나온 비율
- 유효옵션 출현률:
  - 사용자가 선택한 주요 옵션이 2줄 이상 나온 비율
  - 서로 다른 주요 옵션이 각각 1줄씩 나온 경우도 유효옵션으로 인정

예시:

- `DEX 1줄` -> 주요옵션 `True`, 유효옵션 `False`
- `DEX 2줄` -> 주요옵션 `True`, 유효옵션 `True`
- `DEX 1줄 + 공격력 1줄` -> 주요옵션 `True`, 유효옵션 `True`
- `DEX 1줄 + 잡옵 2줄` -> 주요옵션 `True`, 유효옵션 `False`

## 좋았던 조건 / 아쉬웠던 조건 계산 방식

랭킹은 단순 비율 순위가 아니라 표본 수를 반영한 보정 점수로 계산합니다.

```text
adjusted_score = gap_p * log(attempts + 1)
```

- `gap_p`
  - 기준 확률 CSV가 있으면 `기준 확률 대비 차이`
  - 없으면 `전체 평균 대비 차이`
- 시도 수 `n < 10`인 조건은 `참고 불가`로 보고 랭킹에서 제외합니다.
- 큐브는 주요옵션 출현률, 유효옵션 출현률, 등급업률 기준으로 봅니다.
- 스타포스는 성공률이 좋게 관측된 조건과 파괴율이 아쉽게 관측된 조건을 따로 봅니다.

## 신뢰도 기준

- `n < 10`: 참고 불가
- `10 <= n < 30`: 낮음
- `30 <= n < 100`: 보통
- `n >= 100`: 높음

신뢰도는 확률의 진실성을 뜻하지 않습니다. 과거 기록 표본 수 기준의 참고 강도입니다.

## 날짜 범위와 API 제한

- 스타포스 조회 가능 시작일: `2023-12-27`
- 잠재능력 재설정 조회 가능 시작일: `2024-01-25`
- 큐브/잠재능력/스타포스 히스토리는 최대 최근 2년 데이터만 조회할 수 있습니다.

예를 들어 오늘이 `2026-05-04`라면 앱이 계산하는 조회 가능 기간은 다음과 같습니다.

```text
2024-05-05 ~ 2026-05-04
```

앱은 잘못된 날짜가 입력되면 자동으로 이 범위 안으로 보정합니다.

## 스타포스 API 호출 방식

스타포스 강화 결과 조회는 `start_date/end_date`를 직접 받지 않습니다.

- 첫 호출:
  - `/maplestory/v1/history/starforce?date=YYYY-MM-DD&count=1000`
- 다음 페이지:
  - `/maplestory/v1/history/starforce?cursor=next_cursor&count=1000`

여러 날짜를 조회할 때는 앱 내부에서 하루씩 순회하며 호출합니다.

스타포스 응답 핵심 key:

- `count`
- `next_cursor`
- `starforce_history`

## 화면 표시 원칙

- 분석 결과는 Streamlit 화면 안에서 카드, 차트, DataFrame으로만 보여줍니다.
- 이번 버전에서는 분석 결과 CSV나 Excel 파일을 생성하지 않습니다.
- CSV 업로드 분석과 결과 다운로드 기능은 제공하지 않습니다.

## 설치 방법

```bash
cd maple_luck_report
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 실행 방법

```bash
streamlit run app.py
```

## .env 설정

`.env.example`을 참고해 프로젝트 루트에 `.env` 파일을 둘 수 있습니다.

```bash
NEXON_OPEN_API_KEY=발급받은_API_KEY
ENABLE_ANALYTICS=true
ENABLE_LOCAL_STATE_PERSISTENCE=false
POSTHOG_API_KEY=
POSTHOG_HOST=https://app.posthog.com
APP_VERSION=1.0.0
```

기본 동작은 Streamlit 사이드바의 API Key 입력이며, `.env`는 로컬 개발 편의를 위한 선택 사항입니다.

## Streamlit Cloud Secrets

Streamlit Cloud에서는 `.env` 대신 `App -> Settings -> Secrets`에 아래 값을 넣을 수 있습니다.

```toml
ENABLE_ANALYTICS = "true"
ENABLE_LOCAL_STATE_PERSISTENCE = "false"
POSTHOG_API_KEY = "phc_xxxxxxxxx"
POSTHOG_HOST = "https://app.posthog.com"
APP_VERSION = "1.0.0"
```

PostHog API Key가 없으면 내부 SQLite 기반 analytics만 동작하고, PostHog 전송은 자동으로 비활성화됩니다.

## 다중 사용자 배포 안전성

- 기본값으로 서버 파일 기반 상태 복원은 비활성화되어 있습니다.
- 즉, `.app_state/last_state.pkl`를 사용해 이전 사용자의 캐릭터/기록/옵션을 다음 사용자에게 복원하지 않습니다.
- 사용자별 분석 데이터와 화면 상태는 Streamlit `session_state` 안에서만 유지됩니다.
- `ENABLE_LOCAL_STATE_PERSISTENCE=true`를 명시적으로 켜지 않는 한 `.app_state/last_state.pkl`는 생성되지 않습니다.
- Streamlit Cloud 같은 공유 배포 환경에서는 로컬 상태 저장을 강제로 비활성화합니다.
- 기존 로컬 개발 과정에서 생성된 `.app_state/last_state.pkl`가 있다면 삭제 대상으로 보고 정리하는 것을 권장합니다.

## PostHog 집계 요약 이벤트

- `analysis_summary_generated` 이벤트는 사용자가 기록을 불러와 분석이 계산된 뒤 집계 통계량만 전송합니다.
- 예시 전송 항목:
  - `date_range_days`
  - `cube_attempts`, `potential_attempts`, `starforce_attempts`
  - `major_option_rate`, `effective_option_rate`, `grade_up_rate`
  - `starforce_success_rate`, `starforce_destroy_rate`
  - `character_class`, `world_name`, `character_level_bucket`
  - `best_cube_day_of_month`, `best_cube_hour`, `best_cube_weekday`, `best_cube_type`
  - `best_starforce_day_of_month`, `best_starforce_hour`, `best_starforce_weekday`, `best_starforce_transition`
- 원본 큐브/스타포스 기록, API Key 원문, 캐릭터명 원문, ocid 원문은 PostHog로 전송하지 않습니다.

## API 디버그 탭

디버그 탭에서는 다음을 확인할 수 있습니다.

- API Key 설정 여부
- 마지막 동기화 시각
- 수집 기간
- 큐브 기록 수
- 스타포스 기록 수
- 마지막 호출 `path`
- 마지막 호출 `params`
- `status_code`
- `response_keys`
- `count`
- `record_key_exists`
- `record_count`
- `next_cursor_exists`
- 캐시 재사용 날짜 수
- 실제 API 호출 날짜 수
- 실패한 날짜 목록

API Key 자체는 화면에 표시하지 않습니다. 운영 로그와 PostHog 전송에도 API Key 원문, 캐릭터명 원문, ocid 원문, raw API 응답과 원본 기록은 포함하지 않습니다. 내부 `analytics.db`에는 익명 이벤트 로그만 저장되며, 사용자의 원본 분석 데이터는 저장하지 않습니다.

## 주의사항

- 본 서비스는 Nexon Open API로 조회 가능한 본인 히스토리를 기반으로 한 개인 통계 리포트입니다.
- 모든 결과는 과거 기록상 어떻게 관측되었는지를 보여주는 참고용 분석입니다.
- 표본 수가 적은 조건은 참고용으로만 해석해야 합니다.
- 기준 확률 대비 높게 관측되었다는 사실이 향후 결과를 보장하지 않습니다.
- 이 앱은 미래 성공을 예측하거나 보장하지 않습니다.
- API Key는 저장하지 않으며, 사용자의 세션에서만 사용합니다.
- PostHog를 사용할 경우에도 익명 사용자 ID 기준 이벤트만 전송하며, 민감한 원본 데이터는 전송하지 않습니다.
- 다중 사용자 공유 배포에서는 사용자 A의 캐릭터/기록/옵션이 사용자 B에게 복원되지 않도록 서버 파일 기반 상태 저장을 기본 비활성화합니다.

## 다중 사용자 확인 방법

1. 브라우저 A에서 캐릭터와 기록을 불러옵니다.
2. 다른 브라우저 또는 시크릿 창 B로 같은 앱에 접속합니다.
3. B 화면은 빈 초기 상태로 시작해야 하며, A의 캐릭터명/기록 수/옵션/분석 결과가 보이면 안 됩니다.
4. A와 B는 각각 별도의 `session_state` 안에서만 상태를 유지합니다.

## 한계점

- 전체 유저 평균 데이터는 제공되지 않으므로 개인 기록 내부 평균 비교가 중심입니다.
- 직업 정보와 캐릭터 이미지는 Open API 응답에 따라 보강 가능하지만, 현재는 기록 데이터 중심으로 표시합니다.
- 이벤트 달력, 추가적인 공식 확률표, 캐릭터 프로필 확장은 추후 개선 항목입니다.
