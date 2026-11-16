"""
Bowenpally Real Estate Intelligence Dashboard
Run:  streamlit run app.py
Needs bowenpally_house_data.csv in the same folder (or upload it in the sidebar).
"""
import hashlib
from datetime import datetime
from pathlib import Path

import altair as alt
import folium
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from branca.colormap import LinearColormap
from scipy.spatial import ConvexHull

st.set_page_config(page_title="Bowenpally Real Estate", page_icon="🏘️", layout="wide")

# ---------- STYLE ----------
st.markdown("""
<style>
.block-container {padding-top: 1.5rem;}
.kpi {background: linear-gradient(135deg,#2c3e50,#34495e); color:white; padding:16px 18px;
      border-radius:12px; box-shadow:0 2px 8px rgba(0,0,0,.15);}
.kpi .lbl {font-size:12px; opacity:.75; text-transform:uppercase; letter-spacing:1px;}
.kpi .val {font-size:24px; font-weight:700; margin-top:4px;}
.kpi .sub {font-size:12px; opacity:.8; margin-top:2px;}
.kpi.green {background: linear-gradient(135deg,#1e8449,#27ae60);}
.kpi.orange {background: linear-gradient(135deg,#d35400,#f39c12);}
.kpi.blue {background: linear-gradient(135deg,#1f618d,#2980b9);}
</style>
""", unsafe_allow_html=True)

REGION_CENTERS = {
    "RTC Colony": (17.4765, 78.4815), "Sai Nagar": (17.4810, 78.4755),
    "Krishna Nagar": (17.4855, 78.4795), "Raja Rajeshwari Nagar": (17.4811, 78.4777),
    "Ashok Gardens": (17.4785, 78.4855), "Gandhi Nagar": (17.4805, 78.4705),
    "Venkatapuram": (17.4921, 78.5055), "P&T Colony": (17.4725, 78.4985),
    "Indira Nagar": (17.4695, 78.4885), "Hasmathpet Road Area": (17.4754, 78.4857),
}
DEFAULT_CENTER = (17.4761, 78.4828)


def inr(n):
    """Indian digit grouping: 12345678 -> ₹1,23,45,678"""
    s = str(int(n))
    if len(s) <= 3:
        return "₹" + s
    last3, rest = s[-3:], s[:-3]
    groups = []
    while len(rest) > 2:
        groups.insert(0, rest[-2:])
        rest = rest[:-2]
    if rest:
        groups.insert(0, rest)
    return "₹" + ",".join(groups) + "," + last3


def inr_short(n):
    n = float(n)
    if n >= 1e7:
        return f"₹{n/1e7:.2f} Cr"
    if n >= 1e5:
        return f"₹{n/1e5:.1f} L"
    return inr(n)


def coords(row):
    lat, lon = REGION_CENTERS.get(row["Locality"], DEFAULT_CENTER)
    h = hashlib.md5(str(row["Full_Address"]).encode()).hexdigest()
    return (lat + (int(h[:4], 16) / 65535 - 0.5) * 0.0028,
            lon + (int(h[4:8], 16) / 65535 - 0.5) * 0.0028)


def show_chart(chart):
    """Works on both old and new Streamlit versions."""
    try:
        st.altair_chart(chart, width="stretch")
    except TypeError:
        st.altair_chart(chart, use_container_width=True)


def show_table(data, **kw):
    try:
        st.dataframe(data, hide_index=True, width="stretch", **kw)
    except TypeError:
        st.dataframe(data, hide_index=True, use_container_width=True, **kw)


def show_html(html, height):
    if hasattr(st, "iframe"):
        try:
            st.iframe(html, height=height)
            return
        except Exception:
            pass
    components.html(html, height=height)


# Columns the app already knows about; everything else is treated as an amenity column
BASE_COLS = {"Locality", "Street", "Full_Address", "House_Type", "Builtup_Area_sqft",
             "Bedrooms", "Bathrooms", "House_Age_years", "Total_Price_INR",
             "Lat", "Lon", "Future_Price", "Price_per_sqft"}
YES = {"yes", "y", "true", "1", "available", "present"}
NO = {"no", "n", "false", "0", "not available", "none", "na", "n/a", "absent"}


def pretty(col):
    name = col.replace("_", " ").strip()
    for suf in (" km", " m"):
        if name.lower().endswith(suf):
            name = name[: -len(suf)]
    return name.strip().title()


def unit_of(col):
    c = col.lower()
    if c.endswith("_km") or c.endswith(" km"):
        return " km"
    if c.endswith("_m") or c.endswith(" m"):
        return " m"
    return " km"  # distance columns without a unit are assumed to be km


def split_items(val):
    return [x.strip() for x in str(val).replace(";", ",").split(",") if x.strip()]


def classify(df):
    """Sort extra columns into: dist (numeric distances), flag (yes/no),
    multi (comma-separated lists), cat (short text), num (other numbers)."""
    kinds = {"dist": [], "flag": [], "multi": [], "cat": [], "num": []}
    for c in df.columns:
        if c in BASE_COLS:
            continue
        s = df[c].dropna()
        if s.empty:
            continue
        if pd.api.types.is_bool_dtype(df[c]):
            kinds["flag"].append(c)
        elif pd.api.types.is_numeric_dtype(df[c]):
            if "dist" in c.lower() or c.lower().endswith(("_km", "_m")):
                kinds["dist"].append(c)
            elif set(s.unique()) <= {0, 1}:
                kinds["flag"].append(c)
            else:
                kinds["num"].append(c)
        else:
            vals = {str(v).strip().lower() for v in s.unique()}
            if vals <= (YES | NO):
                kinds["flag"].append(c)
            elif s.astype(str).str.contains(r"[,;]").mean() > 0.3:
                kinds["multi"].append(c)
            else:
                kinds["cat"].append(c)
    return kinds


def is_yes(v):
    return str(v).strip().lower() in YES


def fmt_amenity(col, v, kinds):
    if pd.isna(v):
        return None
    if col in kinds["dist"]:
        return f"{v:g}{unit_of(col)}"
    if col in kinds["flag"]:
        return "✅ Yes" if is_yes(v) else "❌ No"
    return str(v)


@st.cache_data
def load(src):
    df = pd.read_csv(src)
    ll = df.apply(lambda r: pd.Series(coords(r)), axis=1)
    df["Lat"], df["Lon"] = ll[0], ll[1]
    return df


def kpi(label, value, sub="", cls=""):
    st.markdown(f'<div class="kpi {cls}"><div class="lbl">{label}</div>'
                f'<div class="val">{value}</div><div class="sub">{sub}</div></div>',
                unsafe_allow_html=True)


# ---------- DATA ----------
st.sidebar.title("🏘️ Filters")
csv_path = Path(__file__).parent / "bowenpally_house_data.csv"
uploaded = st.sidebar.file_uploader("Data file (optional)", type="csv")
if uploaded is not None:
    df = load(uploaded)
elif csv_path.exists():
    df = load(csv_path)
else:
    st.title("🏘️ Bowenpally Real Estate Intelligence")
    st.warning("`bowenpally_house_data.csv` not found. Place it next to app.py "
               "or upload it from the sidebar.")
    st.stop()

kinds = classify(df)
AMEN_COLS = [c for k in kinds.values() for c in k]

# ---------- SIDEBAR FILTERS ----------
localities = st.sidebar.multiselect("Colony", sorted(df["Locality"].unique()),
                                    placeholder="All colonies")
types = st.sidebar.multiselect("House type", sorted(df["House_Type"].unique()),
                               placeholder="All types")
bhk = st.sidebar.multiselect("BHK", sorted(df["Bedrooms"].unique()), placeholder="Any")
pmin, pmax = int(df["Total_Price_INR"].min()), int(df["Total_Price_INR"].max())
price = st.sidebar.slider("Price range", pmin, pmax, (pmin, pmax),
                          step=max((pmax - pmin) // 100, 1000), format="₹%d")
amin, amax = int(df["Builtup_Area_sqft"].min()), int(df["Builtup_Area_sqft"].max())
area = st.sidebar.slider("Built-up area (sqft)", amin, amax, (amin, amax))

# --- amenity filters (built automatically from the dataset's columns) ---
dist_limits, flag_req, cat_sel, multi_sel = {}, [], {}, {}
if AMEN_COLS:
    with st.sidebar.expander("🏥 Amenity filters", expanded=True):
        for c in kinds["dist"]:
            lo, hi = float(df[c].min()), float(df[c].max())
            if hi > lo:
                dist_limits[c] = (lo, hi, st.slider(
                    f"{pretty(c)} (max{unit_of(c)})", lo, hi, hi,
                    step=round((hi - lo) / 100, 3) or 0.01))
        if kinds["flag"]:
            flag_req = st.multiselect("Must have", kinds["flag"], format_func=pretty,
                                      placeholder="Any amenity")
        for c in kinds["cat"]:
            opts = sorted(df[c].dropna().astype(str).unique())
            if 1 < len(opts) <= 25:
                cat_sel[c] = st.multiselect(pretty(c), opts, placeholder="Any")
        for c in kinds["multi"]:
            items = sorted({i for v in df[c].dropna() for i in split_items(v)})
            multi_sel[c] = st.multiselect(f"{pretty(c)} (must include)", items, placeholder="Any")

st.sidebar.divider()
st.sidebar.subheader("📈 Forecast")
years = st.sidebar.slider("Years ahead", 1, 25, 5)
growth = st.sidebar.slider("Annual appreciation (%)", 1.0, 15.0, 7.0, 0.5)
st.sidebar.caption("Forecast is a simple compound-growth assumption, not a market prediction.")

f = df.copy()
if localities: f = f[f["Locality"].isin(localities)]
if types: f = f[f["House_Type"].isin(types)]
if bhk: f = f[f["Bedrooms"].isin(bhk)]
f = f[f["Total_Price_INR"].between(*price) & f["Builtup_Area_sqft"].between(*area)]
for c, (lo, hi, chosen) in dist_limits.items():
    if chosen < hi:                       # only filter when the slider was moved
        f = f[f[c] <= chosen]
for c in flag_req:
    f = f[f[c].map(is_yes)]
for c, sel in cat_sel.items():
    if sel: f = f[f[c].astype(str).isin(sel)]
for c, sel in multi_sel.items():
    if sel: f = f[f[c].map(lambda v: all(i in split_items(v) for i in sel) if pd.notna(v) else False)]
f = f.copy()
f["Future_Price"] = (f["Total_Price_INR"] * (1 + growth / 100) ** years).astype(np.int64)
f["Price_per_sqft"] = (f["Total_Price_INR"] / f["Builtup_Area_sqft"]).round(0)

# ---------- HEADER ----------
st.title("🏘️ Bowenpally Real Estate Intelligence")
st.caption(f"{len(f)} of {len(df)} properties match your filters")

if f.empty:
    st.info("No properties match these filters. Try widening them.")
    st.stop()

year_label = datetime.now().year + years


def find_col(cols, word):
    return next((c for c in cols if word in c.lower()), None)


school_d = find_col(kinds["dist"], "school")
hosp_d = find_col(kinds["dist"], "hospital")

k = st.columns(5)
with k[0]: kpi("Properties", f"{len(f)}", f"{f['Locality'].nunique()} colonies", "blue")
with k[1]: kpi("Avg price", inr_short(f["Total_Price_INR"].mean()), f"Median {inr_short(f['Total_Price_INR'].median())}")
with k[2]: kpi("Avg ₹ / sqft", inr(f["Price_per_sqft"].mean()), f"Avg area {f['Builtup_Area_sqft'].mean():.0f} sqft")
with k[3]: kpi("Nearest school (avg)",
               f"{f[school_d].mean():.2f}{unit_of(school_d)}" if school_d else "n/a",
               "average distance" if school_d else "no school column found", "green")
with k[4]: kpi("Nearest hospital (avg)",
               f"{f[hosp_d].mean():.2f}{unit_of(hosp_d)}" if hosp_d else "n/a",
               "average distance" if hosp_d else "no hospital column found", "orange")
st.write("")

tab_map, tab_amen, tab_charts, tab_data = st.tabs(
    ["🗺️ Map", "🏥 Schools, Hospitals & Amenities", "📊 Analytics", "📋 Properties"])

# ---------- MAP ----------
with tab_map:
    color_options = ["Price"] + [f"Distance: {pretty(c)}" for c in kinds["dist"]]
    mode = st.radio("Colour houses by", color_options, horizontal=True)
    if mode == "Price":
        col_key, vals = "Total_Price_INR", df["Total_Price_INR"]
        cmap = LinearColormap(["#27ae60", "#f39c12", "#d63031"], vmin=vals.min(), vmax=vals.max(),
                              caption="Price (INR): green = budget, red = premium")
    else:
        col_key = kinds["dist"][color_options.index(mode) - 1]
        vals = df[col_key]
        cmap = LinearColormap(["#27ae60", "#f39c12", "#d63031"], vmin=vals.min(), vmax=vals.max(),
                              caption=f"{pretty(col_key)}{unit_of(col_key)}: green = near, red = far")

    center = [f["Lat"].mean(), f["Lon"].mean()]
    zoom = 14 if f["Locality"].nunique() > 1 else 17
    m = folium.Map(location=center, zoom_start=zoom, tiles="OpenStreetMap")
    m.add_child(cmap)

    for loc, g in f.groupby("Locality"):
        if len(g) >= 3:
            pts = g[["Lat", "Lon"]].values
            try:
                hull = ConvexHull(pts)
                folium.Polygon([pts[v].tolist() for v in hull.vertices], color="#2980b9",
                               weight=2, fill=True, fill_opacity=0.07,
                               tooltip=f"{loc} ({len(g)} properties)").add_to(m)
            except Exception:
                pass

    for _, r in f.iterrows():
        rows = ""
        for c in AMEN_COLS:
            val = fmt_amenity(c, r[c], kinds)
            if val:
                rows += f'<tr><td>{pretty(c)}</td><td align="right"><b>{val}</b></td></tr>'
        amen_html = (f'<div style="margin-top:8px; padding:8px; background:#eef6fb; border-radius:6px;">'
                     f'<b>📍 Nearby &amp; amenities</b><table style="width:100%; font-size:12px;">{rows}</table></div>'
                     ) if rows else ""
        pcolor = cmap(r[col_key]) if pd.notna(r[col_key]) else "#95a5a6"
        popup = f"""
        <div style="font-family:Arial; width:290px;">
          <div style="background:{pcolor}; color:white; padding:10px; text-align:center; border-radius:6px 6px 0 0;">
            <b>{r['Full_Address']}</b>
          </div>
          <div style="padding:10px; border:1px solid #eee; border-top:none; font-size:13px;">
            <table style="width:100%;">
              <tr><td><b>Type</b></td><td align="right">{r['House_Type']}</td></tr>
              <tr><td><b>Area</b></td><td align="right">{r['Builtup_Area_sqft']} sqft</td></tr>
              <tr><td><b>Config</b></td><td align="right">{r['Bedrooms']} BHK | {r['Bathrooms']} Bath</td></tr>
              <tr><td><b>Age</b></td><td align="right">{r['House_Age_years']} yrs</td></tr>
              <tr><td><b>₹/sqft</b></td><td align="right">{inr(r['Price_per_sqft'])}</td></tr>
            </table>
            <hr style="margin:8px 0;">
            <div style="color:#27ae60;"><b>Now: {inr(r['Total_Price_INR'])}</b></div>
            <div style="color:#d35400;"><b>{year_label}: {inr(r['Future_Price'])}</b></div>
            {amen_html}
          </div>
        </div>"""
        folium.CircleMarker([r["Lat"], r["Lon"]], radius=7, color="white", weight=2,
                            fill=True, fill_color=pcolor, fill_opacity=1,
                            popup=folium.Popup(popup, max_width=340),
                            tooltip=f"{r['Locality']} · {inr_short(r['Total_Price_INR'])}").add_to(m)

    show_html(m._repr_html_(), 620)

# ---------- AMENITIES ----------
with tab_amen:
    if not AMEN_COLS:
        st.info("No extra amenity columns were found in your CSV. The app treats any column "
                "other than the standard property columns as an amenity.")
        st.write("Columns in your file:", ", ".join(df.columns))
    else:
        with st.expander("Amenity columns detected in your dataset"):
            labels = {"dist": "Distances", "flag": "Yes / No amenities", "multi": "Lists",
                      "cat": "Text / categories", "num": "Other numbers"}
            for key, lab in labels.items():
                if kinds[key]:
                    st.write(f"**{lab}:** " + ", ".join(kinds[key]))

        # ---- nearest facilities for one chosen property ----
        st.subheader("🔎 Check a specific property")
        pick = st.selectbox("Property", f.sort_values(["Locality", "Full_Address"])["Full_Address"],
                            label_visibility="collapsed", key="prop_pick")
        row = f[f["Full_Address"] == pick].iloc[0]
        cards = st.columns(min(max(len(AMEN_COLS), 1), 4))
        for i, c in enumerate(AMEN_COLS):
            v = fmt_amenity(c, row[c], kinds)
            if v:
                with cards[i % len(cards)]:
                    cls = "green" if c in kinds["dist"] and ("school" in c.lower()) else \
                          "orange" if c in kinds["dist"] and ("hospital" in c.lower()) else \
                          "blue" if c in kinds["dist"] else ""
                    kpi(pretty(c), v, row["Locality"] if i == 0 else "", cls)
        st.write("")

        # ---- distance analysis ----
        if kinds["dist"]:
            st.subheader("📏 Distance analysis")
            dcol = st.selectbox("Facility", kinds["dist"], format_func=lambda c: pretty(c),
                                index=kinds["dist"].index(school_d) if school_d else 0)
            a1, a2 = st.columns(2)
            by = f.groupby("Locality")[dcol].mean().reset_index().sort_values(dcol)
            with a1:
                st.markdown(f"**Average {pretty(dcol).lower()} distance by colony**")
                show_chart(alt.Chart(by).mark_bar(cornerRadiusEnd=4, color="#16a085").encode(
                    x=alt.X(f"{dcol}:Q", title=f"Distance ({unit_of(dcol).strip()})"),
                    y=alt.Y("Locality:N", sort="x", title=None),
                    tooltip=["Locality", alt.Tooltip(f"{dcol}:Q", format=".2f")]).properties(height=320))
            with a2:
                st.markdown(f"**Price vs. {pretty(dcol).lower()} distance**")
                show_chart(alt.Chart(f).mark_circle(size=70, opacity=0.75).encode(
                    x=alt.X(f"{dcol}:Q", title=f"Distance ({unit_of(dcol).strip()})"),
                    y=alt.Y("Total_Price_INR:Q", title="Price (INR)"),
                    color=alt.Color("Locality:N", legend=alt.Legend(orient="bottom")),
                    tooltip=["Full_Address", alt.Tooltip(f"{dcol}:Q", format=".2f"),
                             alt.Tooltip("Total_Price_INR:Q", format=",.0f")]
                ).properties(height=320).interactive())

            st.markdown(f"**10 properties closest to {pretty(dcol).lower()}**")
            near = f.nsmallest(10, dcol)[["Full_Address", "Locality", dcol, "Total_Price_INR"]]
            show_table(near, column_config={
                dcol: st.column_config.NumberColumn(f"{pretty(dcol)}{unit_of(dcol)}", format="%.2f"),
                "Total_Price_INR": st.column_config.NumberColumn("Price", format="₹%d")})

        # ---- availability of yes/no amenities by colony ----
        if kinds["flag"]:
            st.subheader("✅ Amenity availability by colony (% of properties)")
            pct = f.groupby("Locality")[kinds["flag"]].agg(lambda s: round(s.map(is_yes).mean() * 100))
            pct.columns = [pretty(c) for c in pct.columns]
            show_table(pct.reset_index(), column_config={
                c: st.column_config.ProgressColumn(c, min_value=0, max_value=100, format="%d%%")
                for c in pct.columns})

# ---------- CHARTS ----------
with tab_charts:
    c1, c2 = st.columns(2)
    by_loc = (f.groupby("Locality")
                .agg(Avg_Price=("Total_Price_INR", "mean"),
                     Avg_PPSF=("Price_per_sqft", "mean"),
                     Count=("Locality", "size"))
                .reset_index().sort_values("Avg_Price", ascending=False))
    with c1:
        st.subheader("Average price by colony")
        show_chart(alt.Chart(by_loc).mark_bar(cornerRadiusEnd=4, color="#2980b9").encode(
            x=alt.X("Avg_Price:Q", title="Avg price (INR)"),
            y=alt.Y("Locality:N", sort="-x", title=None),
            tooltip=["Locality", alt.Tooltip("Avg_Price:Q", format=",.0f"), "Count"]).properties(height=320))
    with c2:
        st.subheader("Price per sqft by colony")
        show_chart(alt.Chart(by_loc).mark_bar(cornerRadiusEnd=4, color="#27ae60").encode(
            x=alt.X("Avg_PPSF:Q", title="Avg ₹ / sqft"),
            y=alt.Y("Locality:N", sort="-x", title=None),
            tooltip=["Locality", alt.Tooltip("Avg_PPSF:Q", format=",.0f")]).properties(height=320))

    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Price vs. built-up area")
        show_chart(alt.Chart(f).mark_circle(size=70, opacity=0.75).encode(
            x=alt.X("Builtup_Area_sqft:Q", title="Area (sqft)"),
            y=alt.Y("Total_Price_INR:Q", title="Price (INR)"),
            color=alt.Color("Locality:N", legend=alt.Legend(orient="bottom")),
            tooltip=["Full_Address", "House_Type", "Builtup_Area_sqft",
                     alt.Tooltip("Total_Price_INR:Q", format=",.0f")]).properties(height=340).interactive())
    with c4:
        st.subheader(f"Value growth over {years} years")
        yrs = np.arange(0, years + 1)
        gdf = pd.DataFrame({"Year": datetime.now().year + yrs,
                            "Avg price": f["Total_Price_INR"].mean() * (1 + growth / 100) ** yrs})
        show_chart(alt.Chart(gdf).mark_area(line={"color": "#d35400"}, color="#f39c12", opacity=0.4).encode(
            x=alt.X("Year:O"), y=alt.Y("Avg price:Q", title="Avg price (INR)"),
            tooltip=["Year", alt.Tooltip("Avg price:Q", format=",.0f")]).properties(height=340))

    st.subheader(f"Forecast: average value in {year_label}")
    st.caption(f"Today {inr_short(f['Total_Price_INR'].mean())}  →  {year_label} "
               f"{inr_short(f['Future_Price'].mean())} at {growth:.1f}% per year")

    st.subheader("Colony summary")
    summ = by_loc.copy()
    summ["Avg_Price"] = summ["Avg_Price"].map(inr)
    summ["Avg_PPSF"] = summ["Avg_PPSF"].map(inr)
    show_table(summ.rename(columns={"Avg_Price": "Avg price", "Avg_PPSF": "Avg ₹/sqft",
                                    "Count": "Properties"}))

# ---------- TABLE ----------
with tab_data:
    cols = ["Full_Address", "Locality", "House_Type", "Bedrooms", "Bathrooms",
            "Builtup_Area_sqft", "House_Age_years"] + AMEN_COLS + \
           ["Total_Price_INR", "Price_per_sqft", "Future_Price"]
    show_table(
        f[cols].sort_values("Total_Price_INR"),
        column_config={
            "Total_Price_INR": st.column_config.NumberColumn("Price", format="₹%d"),
            "Future_Price": st.column_config.NumberColumn(f"Est. {year_label}", format="₹%d"),
            "Price_per_sqft": st.column_config.NumberColumn("₹/sqft", format="₹%d"),
            "Builtup_Area_sqft": st.column_config.NumberColumn("Area (sqft)"),
            "House_Age_years": st.column_config.NumberColumn("Age (yrs)"),
        })
    st.download_button("⬇️ Download filtered data (CSV)",
                       f[cols].to_csv(index=False).encode(), "bowenpally_filtered.csv", "text/csv")