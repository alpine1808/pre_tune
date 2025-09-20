# pre_tune (Gemini-dominant)

Pipeline làm sạch chunk PDF tiếng Việt sau khi được xử lí bởi Simba

## Cấu trúc chính
- `documents` (Postgres): cột `data JSONB` có `filename` và `id` (document id).
- `chunks` (Postgres): `id, document_id, page, offset, text, metadata(JSONB)` — `metadata` có thể chứa `type`, `bbox`, `continues_flag`...

## Cài đặt
```bash
pip install -r requirements.txt
```

## Thiết lập môi trường
Tạo file `.env` từ mẫu `.env.example` hoặc xuất biến môi trường tương ứng.

Các biến dùng được:
- `DATABASE_URL` — ví dụ `postgresql+psycopg://user:pass@localhost:5432/db`
- `GEMINI_API_KEY` — khi dùng Gemini Developer API
- `PRE_TUNE_DRY_RUN` — `1` (mặc định) để **không gọi API**; `0` để gọi API thật
- `GOOGLE_GENAI_USE_VERTEXAI` — `true/false` (mặc định `false`)
- (Vertex) `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION` (vd `us-central1`)
- Tuỳ chọn đổi model: `PRE_TUNE_MODEL_TEXT`, `PRE_TUNE_MODEL_VISION`

## Chạy
```bash
# Dry-run (NO-OP an toàn — không gọi Gemini):
python pre_tune.py --filename "ten_file.pdf" --json-out out.json

# Gọi Gemini thật (Developer API):
export GEMINI_API_KEY="YOUR_KEY"
export PRE_TUNE_DRY_RUN=0
python pre_tune.py --filename "ten_file.pdf" --json-out out.json

# (Tùy chọn) Dùng Vertex:
export GOOGLE_GENAI_USE_VERTEXAI=true
export GOOGLE_CLOUD_PROJECT="your-project-id"
export GOOGLE_CLOUD_LOCATION="us-central1"
export PRE_TUNE_DRY_RUN=0
python pre_tune.py --filename "ten_file.pdf" --json-out out.json
```

## Luồng xử lý
`triage → text_clean (Gemini) → vision_fix (Gemini Vision, gated) → merge_continues → dedupe → disambiguation → locale_normalize → post_validation → package`

> Các bước `merge_continues`… hiện là **điểm móc** (NO-OP) để bạn hiện thực dần theo yêu cầu dự án.
