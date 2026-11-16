# 🏦 بحث دلالي عربي في أسئلة البنك (Arabic Semantic Search + Reranker)

مشروع Encoder-only: العميل يكتب سؤاله بأي لهجة، والنظام يرجّع أقرب سؤال/إجابة من FAQ البنك حتى لو مفيش كلمات مشتركة.

## المعمارية
```
سؤال المستخدم
   ↓  normalize (تشكيل/همزات/أرقام)
Bi-Encoder (multilingual-e5) + cosine  →  Top 20 مرشح
   ↓
Cross-Encoder (bge-reranker-v2-m3)     →  Top 5 مرتبين
```
وفيه **BM25** كـ baseline عشان نقارن.

## التشغيل
```bash
pip install -r requirements.txt
python evaluate.py              # BM25 vs Dense vs Dense+Reranker
python evaluate.py --no-rerank  # بدون الـ reranker (أخف)
streamlit run app.py            # الواجهة
```
أول تشغيل بينزّل الموديلات من HuggingFace (reranker حوالي 2GB). لو جهازك ضعيف غيّر `RERANKER_MODEL` في `src/config.py`.

## الملفات
| الملف | الوظيفة |
|---|---|
| `data/bank_faq.json` | 30 سؤال FAQ + 3 صيغ اختبار لكل سؤال (مصري/فصحى/خليجي) = 90 query |
| `src/normalize.py` | تطبيع النص العربي |
| `src/search.py` | BM25 + Dense Retriever + Reranker + Pipeline |
| `evaluate.py` | Recall@1/3/5 و MRR، وتفصيل حسب اللهجة |
| `app.py` | واجهة Streamlit بتعرض النتايج قبل وبعد الـ reranker |

## ملاحظات مهمة
- بيانات الـ FAQ **عينة افتراضية** لبنك وهمي، والـ test queries مكتوبة يدويًا. للمشروع النهائي ضيف أسئلة حقيقية (من مواقع بنوك أو من أصحابك) عشان الأرقام تبقى أصدق.
- الأرقام اللي في الـ LinkedIn post لازم تكون من تشغيلك أنت لـ `evaluate.py`.

## أفكار للتطوير
- Hybrid search (BM25 + Dense) بـ Reciprocal Rank Fusion.
- Fine-tune الـ bi-encoder على أزواج (سؤال عامية ← FAQ) بـ MultipleNegativesRankingLoss.
- إضافة threshold: لو أعلى score ضعيف يرد "مش لاقي إجابة" بدل نتيجة غلط.
- زيادة الداتا لـ 200+ FAQ وإضافة لهجات (شامي/مغاربي).
