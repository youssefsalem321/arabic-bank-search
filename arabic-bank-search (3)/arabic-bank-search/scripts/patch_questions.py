"""Patch the wording of a few FAQ questions IN PLACE (keeps everything else, e.g. generated FAQs).
Usage (from the project root):  python scripts/patch_questions.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import DATA_PATH  # noqa: E402

PATCH = {
    3: "كيف أفعّل (أشغّل) بطاقتي الجديدة بعد وصولها إليّ؟",
    18: "كيف أغلق (أسكّر) حسابي نهائيًا وأنهي تعاملي مع البنك؟",
    27: "تلقيت رسالة أو اتصالًا مشبوهًا يطلب بياناتي المصرفية، ماذا أفعل؟",
    28: "كيف أبلّغ عن خصم أو سحب لم أقم به من حسابي؟",
}

path = Path(DATA_PATH)
faqs = json.loads(path.read_text(encoding="utf-8"))
backup = path.with_name(path.stem + ".pre_patch.json")
if not backup.exists():
    backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
done = 0
for f in faqs:
    if f["id"] in PATCH:
        f["question"] = PATCH[f["id"]]
        done += 1
path.write_text("[\n" + ",\n".join(json.dumps(f, ensure_ascii=False) for f in faqs) + "\n]\n", encoding="utf-8")
print(f"patched {done}/{len(PATCH)} questions in {path.name} (total FAQs: {len(faqs)}). Backup: {backup.name}")
