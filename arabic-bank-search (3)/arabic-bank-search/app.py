import streamlit as st

from src.config import K_FINAL, K_RETRIEVE
from src.data import load_data, load_threshold
from src.search import SearchPipeline

st.set_page_config(page_title="بحث ذكي في أسئلة البنك", page_icon="🏦", layout="wide")
st.markdown(
    "<style>.rtl{direction:rtl;text-align:right;line-height:1.9}</style>", unsafe_allow_html=True
)


@st.cache_resource(show_spinner="جاري تحميل الموديلات...")
def get_pipeline():
    faqs, queries = load_data()
    return SearchPipeline(faqs), queries


def fmt(x):
    return f"{x:.3f}" if abs(x) >= 0.01 or x == 0 else f"{x:.2e}"


def card(hit, rank=None, big=False):
    title = f"{rank}. " if rank else ""
    size = "1.25em" if big else "1em"
    st.markdown(
        f'<div class="rtl"><b style="font-size:{size}">{title}{hit["question"]}</b><br>{hit["answer"]}<br>'
        f'<small>{hit["category"]} · score {fmt(hit["score"])}</small></div>',
        unsafe_allow_html=True,
    )


pipe, queries = get_pipeline()

st.title("🏦 بحث دلالي عربي في أسئلة البنك الشائعة")
st.caption("Bi-Encoder للاسترجاع ← Cross-Encoder لإعادة الترتيب ← إجابة واحدة")

if load_threshold() is None:
    st.warning(
        "⚠️ حد الرفض لسه مش محسوب، فالنظام بيرجّع أقرب إجابة دايمًا حتى للأسئلة خارج المجال. "
        "شغّل `python evaluate.py --fast --max-queries 300` وبعدها اعمل Restart للواجهة."
    )

with st.sidebar:
    k_retrieve = st.slider("عدد المرشحين من الـ retriever", 5, 30, K_RETRIEVE)
    k_final = st.slider("عدد النتائج في التفاصيل", 2, 10, K_FINAL)
    rule = load_threshold()
    if rule:
        signal = rule["signal"]
        st.caption(f"قاعدة الرفض من evaluate.py: مقياس {signal}")
    else:
        signal = "reranker"
        st.caption("شغّل evaluate.py الأول عشان يحسب حد الرفض تلقائيًا")
    reject_on = st.checkbox("رفض الأسئلة خارج المجال", value=bool(rule))
    conf_default = float(rule["threshold"]) if rule else 0.0
    reject_default = float(rule.get("reject_threshold", conf_default)) if rule else 0.0
    reject_t = st.number_input("حد الرفض (أقل منه: مش لاقي إجابة)", value=reject_default,
                               step=0.001, format="%.5f")
    conf_t = st.number_input("حد الثقة (أقل منه: إجابة بتحذير)", value=max(conf_default, reject_default),
                             step=0.001, format="%.5f")
    example = st.selectbox("جرّب مثال", [""] + [q["query"] for q in queries[:: max(1, len(queries) // 30)]])

query = st.text_input("اكتب سؤالك (فصحى / مصري / خليجي)", value=example)

if query.strip():
    out = pipe.search(query, k_retrieve, k_final)

    top_score = out["reranked"][0]["score"] if signal == "reranker" else out["top_similarity"]
    st.caption(
        f"مقياس الرفض ({signal}) = {fmt(top_score)} | حد الرفض = {fmt(reject_t)} | حد الثقة = {fmt(conf_t)} | "
        f"الرفض {'مفعّل' if reject_on else 'متوقف'}"
    )
    if reject_on and top_score < reject_t:
        st.warning("مش لاقي إجابة مناسبة لسؤالك في الأسئلة الشائعة. جرّب تصيغه بشكل تاني أو كلّم خدمة العملاء.")
    else:
        st.subheader("✅ الإجابة")
        with st.container(border=True):
            card(out["reranked"][0], big=True)
        if reject_on and top_score < conf_t:
            st.warning("ثقة منخفضة: دي أقرب إجابة لقيتها، لو مش اللي تقصده شوف البدائل أو كلّم خدمة العملاء.")
        elif out["confidence"] == "high":
            st.success("ثقة عالية: الـ Retriever والـ Reranker متفقين.")
        else:
            st.info("ثقة متوسطة: الـ Retriever والـ Reranker اختلفوا، شوف البدائل تحت.")

        with st.expander("نتايج تانية محتملة"):
            for i, h in enumerate(out["reranked"][1:], 2):
                card(h, i)
                st.divider()

    with st.expander("تفاصيل: ترتيب الـ Retriever قبل إعادة الترتيب"):
        for i, h in enumerate(out["retrieved"], 1):
            card(h, i)
            st.divider()
