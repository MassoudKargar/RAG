# PRODUCTION TEST REPORT — Maux RAG API

تاریخ: 2026-09-07 | سرور: hermes | دامین: https://rag.masoudkargar.com
برنچ: `master` | کامیتهای مرتبط: `51f482b` → `310ca58` (۶ کامیت)
وضعیت: ✅ همه تستها پاس

---

## ۱. خلاصه

پایپلاین بزرگسند (چانکینگ + امبدینگ + بازیابی + پاسخ RAG) در پروداکشن دیپلوی و
بهصورت کامل تست شد. همهی کامیتها روی `master` پوش شدند و سرویس `rag-api`
ریاستارت شد.

## ۲. تغییرات پیادهسازیشده (Phase 3–15)

| کامیت | تغییر |
|---|---|
| `51f482b` | چانکر بازنویسی + ادغام چانکینگ در `add_document` + upsert (بدون داپلیکیت) + فیکس contract تحلیل + untrack آرتیفکتها از گیت |
| `2de87a2` | فیکس پاراگرافاسپلیتر (تشخیص سکشن/هدینگ) + فلاش هدینگ در مرز سشن |
| `488309f` | رزولوشن مدل پیشفرض بر اساس پروایدر (OpenRouter → `OPENROUTER_CHAT_MODEL`) |
| `b863a83` | مدل چت → `nvidia/nemotron-3-super-120b-a12b:free` (مدل قبلی در OpenRouter حذف شده بود) |
| `7d0bb4e` | مدل چت → `deepseek/deepseek-v4-flash-0731` |
| `80208db` | lexical re-rank برای بازیابی شناسههای دقیق |
| `310ca58` | افزایش candidate pool ریتریوال (۳۰۰) — رفع عدمبازیابی تستهای دور |

## ۳. تستهای خودکار

```
18 passed (tests/): test_chunking (15) + test_e2e_rag (3) — 1:48s
```

- سند ۵۰۰۰ خطی فارسی → ۶۲۵ چانک (۵۴۵ با چانکر قدیمی، ۶۲۵ با سکشن درست)
- بازیابی facts از ابتدا/وسط/انتهای سند ✔
- re-insert همون document_id → جایگزینی بدون داپلیکیت ✔

## ۴. تست E2E پروداکشن (LIVE)

### آپلود سند بزرگ
`POST /v1/vector_db/documents` — سند ۵۰۰۰ خطی، ۴۰۳KB:
- **۵۴۵ چانک در ۳۵ ثانیه** ✔ (ایندکس اول)
- **۶۲۵ چانک در ۳۷ ثانیه** ✔ (re-ingest بعد از فیکس splitter، با سکشنها)

### بازیابی (search_documents) — بعد از فیکس candidate pool
| کوئری | رتبه چانک درست | متادیتا (سکشن) |
|---|---|---|
| «مقدار TEST-001 چیست؟» | ۰ | فصل اول: قوانین پایه |
| «مقدار TEST-002 چیست؟» | ۰ | فصل دوم: قوانین اجرایی |
| «مقدار TEST-003 چیست؟» | ۰ | فصل سوم: قوانین مالی |
| «مقدار TEST-004 را بگو.» | ۰ | فصل چهارم: قوانین تکمیلی |

### RAG + LLM (analysis/query با use_rag=true)
| کوئری | used_rag_context | confidence | پاسخ LLM |
|---|---|---|---|
| «مقدار TEST-003 چیست؟» | ✅ True | ۰.۸۶ | مقدار TEST-003 برابر با 24680 است. ✔ |
| «مقدار TEST-004 را بگو.» | ✅ True | ۰.۸۶ | مقدار TEST-004 برابر 13579 است. ✔ |

### HTTP خارجی (HTTPS از بیرون، با User-Agent معمولی)
- `GET /v1/health` → 200, healthy, RAG available ✔
- `GET /v1/analysis/status` → MCP configured, RAG configured ✔
- بدون `X-API-Key` → 401 ✔ (fail-closed)
- آپلود خارجی کوچک → 201 ✔ + جستجوی سند → مقدار 4242 پیدا شد ✔

## ۵. مشکلات برخوردشده و راهحلها

1. **Cloudflare 403** — فقط برای User-Agent های مرورگر؛ با curl/UA عادی همهچیز باز است. (رفتار CF، نه باگ ما)
2. **مدل free قبلی در OpenRouter حذف شده** (`nemotron-3-nano-30b-a3b:free` → 404، پولی شده). جایگزین: `nemotron-3-super-120b-a12b:free` و سپس `deepseek/deepseek-v4-flash-0731` (به درخواست مسعود).
3. **بازیابی دقیق شناسهها (TEST-xxx)** — با ۶۲۵ چانک، dense retrieval چانک درست را داخل top-10 نیاورد. راهحل: candidate pool ۳۰۰ + lexical re-rank → همه رتبه ۰.
4. **5xx Nvidia overloaded** — transient؛ fallback کار میکرد و reset counter بعد از موفقیت اضافه شد.

## ۶. امنیت — لیک کلیدها در گیت (مهم!)

کلیدها **در تاریخچهی گیت (کامیت `d70fad3` ریشهی master) لو رفتهاند**:
- `OPENROUTER_API_KEY` (sk-or-…، ۷۳) — **باید revoke شود**
- `OPENAI_API_KEY` (sk-…، ۱۷) — revoke یا غیرفعال
- `RAG_API_KEY` (rag-…، ۵۷) — ✅ **عوض شد** (جدید فقط در `.env`)

اقدامات انجامشده:
- `chroma_db/`، `__pycache__/`، `app/.env` از ایندکس گیت حذف شدند (کامیت `51f482b`)
- `.gitignore` پوشش دارد (`chroma_db/`, `.env`, `__pycache__/`)
- `RAG_API_KEY` چرخانده شد (کلید جدید ۵۸ کاراکتری؛ فقط با آن درخواست ۲۰۰ است، بدون آن ۴۰۱)

باقیمانده برای مسعود:
- [ ] revoke `OPENROUTER_API_KEY` قدیمی + گذاشتن کلید جدید در `/var/rag_app/.env`
- [ ] revoke `OPENAI_API_KEY` قدیمی (یا حذف از `.env`)

## ۷. عقبمانده / TODO

- [ ] تست `chat/completions` و `analysis/query` با کلید جدید OpenRouter (پس از ست شدن در `.env`)
- [ ] (پیشنهاد) endpoint حذف انتخابی (`DELETE /documents/{document_id}`) — فعلاً `clear_collection` کل را پاک میکند
- [ ] (پیشنهاد) بازنویسی تاریخچهی گیت با `filter-repo` برای پاککردن کلیدها از history (ترجیح: فقط rotation؛ بازنویسی history نیاز به force push دارد و همکاری شما)