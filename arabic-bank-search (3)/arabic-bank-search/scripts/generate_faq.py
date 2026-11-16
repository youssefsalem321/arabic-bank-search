"""Generate extra bank FAQ entries with an LLM (offline data preparation only).

Setup (Windows PowerShell):
    pip install groq
    $env:GROQ_API_KEY="your_key"
Run from the project root:
    python scripts/generate_faq.py --target 1000

Each call asks for ~12 FAQs about one sub-topic, validates them, removes duplicates
(exact + semantic), and appends them to data/bank_faq.json (a backup is made first).
Safe to stop (Ctrl+C) and re-run: it saves after every call and continues from the file.
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from src.config import DATA_PATH, RETRIEVER_MODEL  # noqa: E402
from src.normalize import normalize  # noqa: E402

TAXONOMY = {
    "بطاقات": ["بطاقات الخصم", "البطاقات الائتمانية", "البطاقات مسبقة الدفع", "الدفع بدون تلامس",
               "الحد الائتماني", "الرقم السري للبطاقة", "استخدام البطاقة في الخارج",
               "نقاط ومكافآت البطاقة", "التقسيط على البطاقة", "فقدان البطاقة أو سرقتها"],
    "تحويلات": ["التحويل المحلي", "التحويل الدولي", "المستفيدون", "حدود التحويل",
                "التحويل الفوري", "التحويل الخاطئ", "رسوم التحويل", "الدفع عبر رمز QR"],
    "حسابات": ["فتح حساب", "إغلاق حساب", "أنواع الحسابات", "كشف الحساب", "تحديث بيانات العميل",
               "الحساب المشترك", "حسابات الأطفال والشباب", "الحساب الراكد",
               "الحساب بالعملة الأجنبية", "الشهادات والودائع"],
    "قروض": ["القرض الشخصي", "تمويل السيارات", "التمويل العقاري", "السداد المبكر",
             "التأخر في السداد", "إعادة جدولة القرض", "الضمانات والكفلاء", "القسط الشهري"],
    "تطبيق": ["التسجيل في التطبيق", "كلمة المرور", "رمز التحقق", "البصمة والتعرف على الوجه",
              "تحديث التطبيق", "أعطال التطبيق", "الإنترنت البنكي", "المحافظ الإلكترونية"],
    "أمان": ["الرسائل والمكالمات المشبوهة", "عمليات غير مصرح بها", "حماية الحساب",
             "التبليغ عن احتيال", "الاعتراض على عملية", "اختراق الحساب"],
    "رسوم": ["رسوم الحساب", "رسوم البطاقات", "رسوم السحب", "رسوم الخدمات", "الضرائب والدمغة"],
    "فروع": ["مواعيد العمل", "مواقع الفروع", "ماكينات الصراف الآلي", "الإيداع النقدي",
             "الخدمة في العطلات", "حجز موعد في الفرع"],
    "خدمة العملاء": ["الخط الساخن", "تقديم شكوى", "تتبع الطلبات", "خدمات ذوي الاحتياجات الخاصة",
                     "الدردشة والدعم"],
    "ادخار": ["حسابات التوفير", "الودائع لأجل", "شهادات الادخار", "أهداف الادخار"],
    "أعمال": ["حساب الشركات", "أجهزة نقاط البيع", "تحويل الرواتب", "خطابات الضمان"],
}
DIALECTS = {"egy", "msa", "gulf"}

PROMPT = """أنت تكتب قاعدة أسئلة شائعة (FAQ) لبنك تجزئة عربي افتراضي.
المجال: <<CATEGORY>> | الموضوع: <<SUBTOPIC>>
اكتب <<N>> أسئلة شائعة مختلفة تمامًا عن بعضها (زوايا وحالات واقعية مختلفة)، كل سؤال موقف يسأل عنه عميل حقيقي.
القواعد:
- "question": سؤال بالفصحى المبسطة.
- "answer": جملة أو جملتان بالفصحى المبسطة، عامة وآمنة. لا تذكر أرقامًا أو رسومًا أو نسبًا أو أسماء بنوك أو أرقام هواتف محددة (استخدم عبارات مثل "حسب نوع الحساب" و"عبر التطبيق").
- "queries": ثلاث صيغ لنفس السؤال كما يكتبها عميل حقيقي: واحدة باللهجة المصرية (egy)، وواحدة بالفصحى (msa)، وواحدة باللهجة الخليجية (gulf). يجب أن تختلف صياغتها عن نص السؤال قدر الإمكان ولا تنسخ كلماته، وبدون علامات ترقيم زائدة.
- "aliases": صيغتان عاميتان إضافيتان لنفس السؤال (مصرية وخليجية)، قصيرتان وتختلفان عن "queries".
- لا تكرر أي سؤال من هذه القائمة: <<EXISTING>>
أرجع JSON فقط بهذا الشكل:
{"faqs":[{"question":"...","answer":"...","aliases":["...","..."],"queries":[{"text":"...","dialect":"egy"},{"text":"...","dialect":"msa"},{"text":"...","dialect":"gulf"}]}]}"""


def build_prompt(cat, sub, n, existing):
    ex = " | ".join(existing) if existing else "(لا يوجد)"
    return (PROMPT.replace("<<CATEGORY>>", cat).replace("<<SUBTOPIC>>", sub)
            .replace("<<N>>", str(n)).replace("<<EXISTING>>", ex))


def validate(item):
    """Returns (question, answer, [[text, dialect], ...], aliases) or None."""
    if not isinstance(item, dict):
        return None
    q, a = str(item.get("question", "")).strip(), str(item.get("answer", "")).strip()
    if not (8 <= len(q) <= 200 and 15 <= len(a) <= 600):
        return None
    by_dialect = {}
    for x in item.get("queries", []):
        if isinstance(x, dict) and x.get("dialect") in DIALECTS:
            t = str(x.get("text", "")).strip()
            if t and normalize(t) != normalize(q):
                by_dialect.setdefault(x["dialect"], t)
    if len(by_dialect) < 2:
        return None
    aliases = [str(x).strip() for x in item.get("aliases", []) if isinstance(x, str) and x.strip()][:3]
    return q, a, [[t, d] for d, t in by_dialect.items()], aliases


class SemanticDedup:
    """Drops questions whose embedding is almost identical to an existing one."""

    def __init__(self, questions, model_name=RETRIEVER_MODEL, threshold=0.95):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.prefix = "passage: " if "e5" in model_name.lower() else ""
        self.threshold = threshold
        dim = self.model.get_sentence_embedding_dimension()
        self.emb = self._enc(questions) if questions else np.zeros((0, dim), dtype=np.float32)

    def _enc(self, qs):
        return self.model.encode([self.prefix + normalize(q) for q in qs],
                                 normalize_embeddings=True, show_progress_bar=False)

    def keep_mask(self, qs):
        new = self._enc(qs)
        mask, kept = [], []
        for v in new:
            pool = np.vstack([self.emb] + kept) if kept else self.emb
            dup = len(pool) > 0 and float(np.max(pool @ v)) >= self.threshold
            mask.append(not dup)
            if not dup:
                kept.append(v[None, :])
        if kept:
            self.emb = np.vstack([self.emb] + kept)
        return mask


def ask_llm(client, model, prompt):
    for attempt in range(4):
        try:
            r = client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": prompt}],
                temperature=0.9, response_format={"type": "json_object"},
            )
            return json.loads(r.choices[0].message.content)["faqs"]
        except Exception as e:  # network, rate limit, bad JSON...
            wait = 2 ** attempt * 2
            print(f"  retry in {wait}s ({e.__class__.__name__}: {str(e)[:80]})")
            time.sleep(wait)
    return []


def save(path, faqs):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text("[\n" + ",\n".join(json.dumps(f, ensure_ascii=False) for f in faqs) + "\n]\n",
                   encoding="utf-8")
    tmp.replace(path)


def run(args, ask_fn, dedup=None):
    path = Path(args.data)
    faqs = json.loads(path.read_text(encoding="utf-8"))
    backup = path.with_name(path.stem + ".backup.json")
    if not backup.exists():
        backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    seen = {normalize(f["question"]) for f in faqs}
    next_id = max(f["id"] for f in faqs) + 1
    if dedup is None and not args.no_semantic_dedup:
        dedup = SemanticDedup([f["question"] for f in faqs])
    subtopics = [(c, s) for c, subs in TAXONOMY.items() for s in subs]
    added, rounds = 0, 0
    while added < args.target and rounds < args.max_rounds:
        rounds += 1
        random.shuffle(subtopics)
        for cat, sub in subtopics:
            if added >= args.target:
                break
            existing = [f["question"] for f in faqs if f.get("subtopic") == sub][-25:]
            items = ask_fn(build_prompt(cat, sub, args.per_call, existing))
            cands = [v for v in (validate(i) for i in items) if v and normalize(v[0]) not in seen]
            if cands and dedup is not None:
                cands = [c for c, ok in zip(cands, dedup.keep_mask([c[0] for c in cands])) if ok]
            for q, a, queries, aliases in cands[: args.target - added]:
                faqs.append({"id": next_id, "category": cat, "subtopic": sub, "question": q,
                             "answer": a, "q": queries, "source": "llm"})
                if getattr(args, "aliases", False) and aliases:
                    faqs[-1]["aliases"] = aliases
                seen.add(normalize(q))
                next_id += 1
                added += 1
            save(path, faqs)
            print(f"[round {rounds}] {cat} / {sub}: +{len(cands)}  (added {added}/{args.target}, total {len(faqs)})")
            time.sleep(args.sleep)
    print(f"\nDone. Added {added} FAQs. Total now {len(faqs)}. Backup: {backup.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=1000, help="how many NEW FAQs to add")
    ap.add_argument("--per-call", type=int, default=12)
    ap.add_argument("--max-rounds", type=int, default=6)
    ap.add_argument("--model", default="llama-3.3-70b-versatile")
    ap.add_argument("--data", default=str(DATA_PATH))
    ap.add_argument("--sleep", type=float, default=1.0)
    ap.add_argument("--no-semantic-dedup", action="store_true")
    ap.add_argument("--aliases", action="store_true",
                    help="also store LLM aliases (off by default: aliases on many FAQs lowered accuracy)")
    args = ap.parse_args()

    if not os.environ.get("GROQ_API_KEY"):
        sys.exit("Set GROQ_API_KEY first (PowerShell: $env:GROQ_API_KEY=\"your_key\")")
    try:
        from groq import Groq
    except ImportError:
        sys.exit("Run: pip install groq")
    client = Groq()
    run(args, lambda p: ask_llm(client, args.model, p))


if __name__ == "__main__":
    main()
