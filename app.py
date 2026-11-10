"""
================================================================================
DỰ ÁN: VN-STOCK SCREENER PRO (TRADINGVIEW & FIREAT STYLE) - BẢN SỬA LỖI & DEBUG
CHỨC NĂNG: Lọc cổ phiếu toàn thị trường theo Minervini SEPA & Livermore Pivot
TÁC GIẢ: Senior Frontend Developer & Pro Stock Trader
================================================================================
"""

import io
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import streamlit as st

# ==============================================================================
# 1. THIẾT LẬP CẤU HÌNH TRANG VÀ GIAO DIỆN DARK PRO
# ==============================================================================
st.set_page_config(
    page_title="VN-Stock Screener Pro | Institutional Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp {
        background-color: #0b0e14;
        color: #d1d4dc;
        font-family: -apple-system, BlinkMacSystemFont, 'Trebuchet MS', Roboto, Ubuntu, sans-serif;
    }
    .header-box {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: linear-gradient(90deg, #131722 0%, #1e222d 100%);
        padding: 16px 24px;
        border-radius: 8px;
        border: 1px solid #2a2e39;
        margin-bottom: 20px;
    }
    .header-title {
        font-size: 24px;
        font-weight: 800;
        color: #2962ff;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .header-desc {
        font-size: 13px;
        color: #787b86;
        margin-top: 4px;
    }
    .badge-live {
        background-color: rgba(14, 203, 129, 0.12);
        color: #0ecb81;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        border: 1px solid rgba(14, 203, 129, 0.3);
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .badge-live::before {
        content: "";
        width: 8px;
        height: 8px;
        background-color: #0ecb81;
        border-radius: 50%;
        box-shadow: 0 0 8px #0ecb81;
    }
    .kpi-container {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 12px;
        margin-bottom: 20px;
    }
    .kpi-card {
        background: #131722;
        border: 1px solid #2a2e39;
        border-radius: 6px;
        padding: 14px 18px;
    }
    .kpi-label {
        font-size: 12px;
        color: #787b86;
        text-transform: uppercase;
        font-weight: 600;
    }
    .kpi-value {
        font-size: 22px;
        font-weight: 700;
        color: #f0f3fa;
        margin-top: 4px;
    }
    [data-testid="stSidebar"] {
        background-color: #131722;
        border-right: 1px solid #2a2e39;
    }
    div.stButton > button {
        background-color: #2a2e39;
        color: #d1d4dc;
        border: 1px solid #363a45;
        border-radius: 6px;
        font-weight: 600;
    }
    div.stButton > button:hover {
        background-color: #2962ff;
        color: #ffffff;
        border-color: #2962ff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# 2. DATA PROVIDER: TRUY XUẤT VÀ KIỂM TRA DỮ LIỆU CHỨNG KHOÁN (TCBS API)
# ==============================================================================
class TCBSDataProvider:
    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json",
    }

    FALLBACK_VN30 = [
        "ACB", "BCM", "BID", "BVH", "CTG", "FPT", "GAS", "GVR", "HDB", "HPG",
        "MBB", "MSN", "MWG", "PLX", "POW", "SAB", "SHB", "SSB", "SSI", "STB",
        "TCB", "TPB", "VCB", "VHM", "VIB", "VIC", "VJC", "VNM", "VPB", "VRE",
    ]

    @staticmethod
    @st.cache_data(ttl=86400, show_spinner=False)
    def fetch_all_listed_tickers() -> pd.DataFrame:
        """Lấy danh sách mã niêm yết có fallback dự phòng an toàn."""
        url = "https://apipubaws.tcbs.com.vn/tcanalysis/v1/finance/all-stock-symbol"
        try:
            res = requests.get(url, headers=TCBSDataProvider.HEADERS, timeout=6)
            if res.status_code == 200:
                data = res.json()
                symbols_raw = data if isinstance(data, list) else data.get("data", [])
                rows = []
                for item in symbols_raw:
                    if isinstance(item, dict):
                        code = item.get("ticker") or item.get("symbol") or item.get("code")
                        floor = item.get("floor") or item.get("exchange") or "HOSE"
                        stype = item.get("type", "stock")
                        if code and len(code) == 3 and str(stype).lower() in ("stock", ""):
                            rows.append({"Symbol": code.upper(), "Exchange": floor.upper()})
                if rows:
                    return pd.DataFrame(rows).drop_duplicates(subset=["Symbol"])
        except Exception:
            pass

        # Fallback danh sách nòng cốt nếu kết nối máy chủ liệt kê bị nghẽn
        return pd.DataFrame([{"Symbol": s, "Exchange": "HOSE"} for s in TCBSDataProvider.FALLBACK_VN30])

    @staticmethod
    @st.cache_data(ttl=300, show_spinner=False)
    def get_price_history(symbol: str, days_back: int = 400) -> pd.DataFrame:
        """Lấy lịch sử giá nến ngày (OHLCV) và chuẩn hóa tên cột."""
        try:
            end_ts = int(time.time())
            start_ts = end_ts - (days_back * 86400)
            url = (
                f"https://apipubaws.tcbs.com.vn/stock-insight/v1/stock/bars-long-term?"
                f"ticker={symbol}&type=stock&resolution=D&from={start_ts}&to={end_ts}"
            )
            res = requests.get(url, headers=TCBSDataProvider.HEADERS, timeout=6)
            if res.status_code != 200:
                return pd.DataFrame()

            payload = res.json().get("data", [])
            if not payload or not isinstance(payload, list):
                return pd.DataFrame()

            df = pd.DataFrame(payload)

            # Chuẩn hóa tên cột đề phòng trường hợp API đổi định dạng viết tắt
            rename_map = {
                "tradingDate": "Date", "d": "Date",
                "open": "Open", "o": "Open",
                "high": "High", "h": "High",
                "low": "Low", "l": "Low",
                "close": "Close", "c": "Close",
                "volume": "Volume", "v": "Volume"
            }
            df.rename(columns=rename_map, inplace=True)

            required_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
            if not all(col in df.columns for col in required_cols):
                return pd.DataFrame()

            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df.dropna(subset=["Date", "Close"], inplace=True)
            df.sort_values("Date", inplace=True)
            df.reset_index(drop=True, inplace=True)

            # Quy đổi đơn vị giá về nghìn đồng nếu giá trị thô > 1000
            if len(df) > 0 and df["Close"].iloc[-1] > 1000:
                df[["Open", "High", "Low", "Close"]] = df[["Open", "High", "Low", "Close"]] / 1000.0

            return df[required_cols]
        except Exception:
            return pd.DataFrame()

    @staticmethod
    @st.cache_data(ttl=3600, show_spinner=False)
    def get_financial_ratios(symbol: str) -> Dict[str, float]:
        """Lấy dữ liệu cơ bản FA (ROE, Tăng trưởng EPS/Doanh thu)."""
        fallback_res = {"roe": 0.0, "eps_growth_yoy": 0.0, "rev_growth_yoy": 0.0, "quarter": "N/A"}
        try:
            url = f"https://apipubaws.tcbs.com.vn/tcanalysis/v1/finance/{symbol}/financialratio?yearly=0&isAll=false"
            res = requests.get(url, headers=TCBSDataProvider.HEADERS, timeout=5)
            if res.status_code != 200:
                return fallback_res

            items = res.json()
            if not isinstance(items, list) or len(items) == 0:
                return fallback_res

            items = sorted(items, key=lambda x: (x.get("year", 0), x.get("quarter", 0)), reverse=True)
            latest = items[0]

            roe_val = float(latest.get("roe", 0) or 0)
            if 0 < roe_val <= 1.0:
                roe_val *= 100

            eps_g, rev_g = 0.0, 0.0
            if len(items) >= 5:
                past = items[4]
                curr_eps = float(latest.get("earningPerShare", 0) or 0)
                past_eps = float(past.get("earningPerShare", 0) or 0)
                if past_eps != 0:
                    eps_g = ((curr_eps - past_eps) / abs(past_eps)) * 100

                curr_rev = float(latest.get("netRevenue", 0) or latest.get("revenue", 0) or 0)
                past_rev = float(past.get("netRevenue", 0) or past.get("revenue", 0) or 0)
                if past_rev != 0:
                    rev_g = ((curr_rev - past_rev) / abs(past_rev)) * 100

            return {
                "roe": round(roe_val, 1),
                "eps_growth_yoy": round(eps_g, 1),
                "rev_growth_yoy": round(rev_g, 1),
                "quarter": f"Q{latest.get('quarter', '?')}/{latest.get('year', '?')}",
            }
        except Exception:
            return fallback_res


# ==============================================================================
# 3. CORE LOGIC: PHÂN TÍCH CỔ PHIẾU (ĐÃ BỌC TOÀN BỘ TRY-EXCEPT VÀ KIỂM TRA ĐỘ DÀI)
# ==============================================================================
def analyze_stock(symbol: str, exchange: str, params: dict) -> Dict:
    """
    Phân tích cổ phiếu theo tiêu chuẩn Minervini SEPA & Livermore Pivot.
    Bọc toàn bộ logic trong khối try-except tổng quát để đảm bảo không crash ứng dụng.
    """
    try:
        # Lấy dữ liệu giá lịch sử
        df = TCBSDataProvider.get_price_history(symbol, days_back=380)

        # KIỂM TRA ĐIỀU KIỆN TIÊN QUYẾT: DataFrame phải hợp lệ và có tối thiểu 50 phiên
        if df.empty or len(df) < 50:
            return {
                "status": "skipped",
                "symbol": symbol,
                "reason": f"Dữ liệu rỗng hoặc chưa đủ 50 phiên (Hiện có: {len(df)} phiên)"
            }

        # 1. Tính toán các đường Trung bình Động (MA)
        df["MA50"] = df["Close"].rolling(window=params["ma_fast"]).mean()
        # Đối với MA150 & MA200, nếu chưa đủ phiên, dùng min_periods để tránh NaN toàn bộ
        df["MA150"] = df["Close"].rolling(window=params["ma_mid"], min_periods=50).mean()
        df["MA200"] = df["Close"].rolling(window=params["ma_slow"], min_periods=50).mean()
        df["MA20_Vol"] = df["Volume"].rolling(window=20, min_periods=5).mean()

        curr = df.iloc[-1]
        prev = df.iloc[-2]

        close_p = float(curr["Close"])
        prev_p = float(prev["Close"])
        change_pct = ((close_p - prev_p) / prev_p) * 100 if prev_p > 0 else 0.0
        vol_curr = float(curr["Volume"])
        ma20_vol = float(curr["MA20_Vol"]) if curr["MA20_Vol"] > 0 else 1.0

        # Kiểm tra ngưỡng thanh khoản tối thiểu
        if ma20_vol < params["min_liquidity_vol"]:
            return {"status": "skipped", "symbol": symbol, "reason": "Thanh khoản dưới mức tối thiểu"}

        # 2. Đỉnh / Đáy 52 tuần
        lookback_len = min(len(df), 250)
        high_52w = float(df["High"].iloc[-lookback_len:].max())
        low_52w = float(df["Low"].iloc[-lookback_len:].min())

        pct_above_low = ((close_p - low_52w) / low_52w) * 100 if low_52w > 0 else 0.0
        pct_below_high = ((high_52w - close_p) / high_52w) * 100 if high_52w > 0 else 0.0

        # 3. Tiêu chí Trend Template (Mark Minervini SEPA)
        ma200_20d_ago = df["MA200"].iloc[-21] if len(df) >= 21 else df["MA200"].iloc[0]
        c_ma_stack = (close_p > curr["MA50"]) and (curr["MA50"] > curr["MA150"]) and (curr["MA150"] > curr["MA200"])
        c_ma200_up = curr["MA200"] > ma200_20d_ago
        c_above_low = pct_above_low >= params["min_from_low"]
        c_near_high = pct_below_high <= params["max_from_high"]
        is_trend_template = c_ma_stack and c_ma200_up and c_above_low and c_near_high

        # 4. Mẫu hình VCP & Cạn Kiệt Volume
        df["H-L"] = df["High"] - df["Low"]
        df["H-PC"] = (df["High"] - df["Close"].shift(1)).abs()
        df["L-PC"] = (df["Low"] - df["Close"].shift(1)).abs()
        df["TR"] = df[["H-L", "H-PC", "L-PC"]].max(axis=1)
        atr5 = float(df["TR"].rolling(5, min_periods=1).mean().iloc[-1])
        atr20 = float(df["TR"].rolling(20, min_periods=1).mean().iloc[-1])

        is_vcp = atr5 < (atr20 * params["vcp_threshold"])
        vol_3d_avg = float(df["Volume"].iloc[-3:].mean())
        is_vol_dryup = vol_3d_avg < (0.8 * ma20_vol)

        # 5. Điểm nổ Livermore Pivot Breakout
        is_pivot = (change_pct >= params["pivot_price_pct"]) and (vol_curr >= (params["pivot_vol_mult"] * ma20_vol))

        # 6. Dữ liệu Cơ bản FA
        fa_info = {"roe": 0.0, "eps_growth_yoy": 0.0, "rev_growth_yoy": 0.0, "quarter": "N/A"}
        if params["use_fa"]:
            fa_info = TCBSDataProvider.get_financial_ratios(symbol)

        pass_fa = True
        if params["use_fa"]:
            pass_fa = (
                fa_info["eps_growth_yoy"] >= params["min_eps_g"]
                and fa_info["rev_growth_yoy"] >= params["min_rev_g"]
                and fa_info["roe"] >= params["min_roe"]
            )

        # 7. Tính điểm kỹ thuật (Thang điểm 100)
        score = 0
        if c_ma_stack: score += 30
        if c_ma200_up: score += 15
        if c_above_low and c_near_high: score += 15
        if is_vcp or is_vol_dryup: score += 15
        if is_pivot: score += 25

        return {
            "status": "success",
            "data": {
                "Symbol": symbol,
                "Exchange": exchange,
                "Price": round(close_p, 2),
                "Change_%": round(change_pct, 2),
                "Volume": int(vol_curr),
                "MA20_Vol": int(ma20_vol),
                "Score": score,
                "Trend": is_trend_template,
                "VCP": (is_vcp or is_vol_dryup),
                "Pivot": is_pivot,
                "Pass_FA": pass_fa,
                "Cách_Đỉnh_%": round(pct_below_high, 1),
                "Trên_Đáy_%": round(pct_above_low, 1),
                "ROE": fa_info["roe"],
                "EPS_G": fa_info["eps_growth_yoy"],
                "Rev_G": fa_info["rev_growth_yoy"],
                "Quarter": fa_info["quarter"],
            }
        }
    except Exception as e:
        # Bắt toàn bộ ngoại lệ và trả về thông tin lỗi để hiển thị ra màn hình debug
        return {
            "status": "error",
            "symbol": symbol,
            "error_msg": str(e)
        }


# ==============================================================================
# 4. PLOTLY CHART ENGINE
# ==============================================================================
def render_tradingview_chart(symbol: str):
    """Vẽ biểu đồ Nến Nhật & Volume chuyên nghiệp."""
    df = TCBSDataProvider.get_price_history(symbol, days_back=260)
    if df.empty or len(df) < 20:
        st.warning(f"Không đủ dữ liệu nến để vẽ biểu đồ cho mã {symbol}.")
        return

    df["MA50"] = df["Close"].rolling(50, min_periods=10).mean()
    df["MA150"] = df["Close"].rolling(150, min_periods=20).mean()
    df["MA200"] = df["Close"].rolling(200, min_periods=30).mean()
    df["MA20_Vol"] = df["Volume"].rolling(20, min_periods=5).mean()

    candle_colors = ["#089981" if c >= o else "#f23645" for o, c in zip(df["Open"], df["Close"])]

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.72, 0.28],
        specs=[[{"type": "candlestick"}], [{"type": "bar"}]],
    )

    fig.add_trace(
        go.Candlestick(
            x=df["Date"], open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
            name="Giá", increasing_line_color="#089981", decreasing_line_color="#f23645",
        ),
        row=1, col=1,
    )

    fig.add_trace(go.Scatter(x=df["Date"], y=df["MA50"], line=dict(color="#2962ff", width=1.5), name="MA50"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["MA150"], line=dict(color="#ff9800", width=1.5), name="MA150"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["MA200"], line=dict(color="#e91e63", width=2), name="MA200"), row=1, col=1)

    fig.add_trace(go.Bar(x=df["Date"], y=df["Volume"], marker_color=candle_colors, name="Volume"), row=2, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["MA20_Vol"], line=dict(color="#ffeb3b", width=1.2), name="MA20 Vol"), row=2, col=1)

    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="#131722",
        paper_bgcolor="#131722",
        height=560,
        margin=dict(l=40, r=40, t=20, b=20),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=11, color="#787b86")),
        yaxis=dict(gridcolor="#2a2e39", title="Giá (k VND)", tickfont=dict(color="#787b86")),
        yaxis2=dict(gridcolor="#2a2e39", title="Khối lượng", tickfont=dict(color="#787b86")),
        xaxis=dict(gridcolor="#2a2e39", tickfont=dict(color="#787b86")),
    )
    st.plotly_chart(fig, use_container_width=True)


# ==============================================================================
# 5. HEADER SECTION
# ==============================================================================
st.markdown(
    """
    <div class="header-box">
        <div>
            <div class="header-title">⚡ VN-STOCK SCREENER PRO</div>
            <div class="header-desc">Hệ thống Lọc Cổ Phiếu Tăng Trưởng Toàn Thị Trường • Minervini SEPA & Livermore Pocket Pivot</div>
        </div>
        <div class="badge-live">KẾT NỐI REALTIME</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# 6. SIDEBAR CẤU HÌNH BỘ LỌC
# ==============================================================================
if "req_trend" not in st.session_state:
    st.session_state.req_trend = True
if "req_pivot" not in st.session_state:
    st.session_state.req_pivot = False
if "req_vcp" not in st.session_state:
    st.session_state.req_vcp = False

with st.sidebar:
    st.markdown("### 🎛️ BỘ ĐIỀU KHIỂN CHIẾN LƯỢC")

    # QUICK PRESETS
    st.markdown("<small style='color:#787b86;'>CHIẾN LƯỢC CÀI ĐẶT NHANH:</small>", unsafe_allow_html=True)
    c_p1, c_p2 = st.columns(2)
    with c_p1:
        if st.button("🏆 SEPA Chuẩn", use_container_width=True):
            st.session_state.req_trend = True
            st.session_state.req_pivot = False
            st.session_state.req_vcp = False
            st.rerun()
    with c_p2:
        if st.button("🚀 Pivot Nổ", use_container_width=True):
            st.session_state.req_trend = True
            st.session_state.req_pivot = True
            st.session_state.req_vcp = False
            st.rerun()

    st.markdown("---")
    exchange_choice = st.selectbox(
        "Sàn Giao Dịch:",
        ["Tất cả (HOSE, HNX, UPCoM)", "HOSE (Sàn TP.HCM)", "HNX (Sàn Hà Nội)", "UPCoM", "Top Cổ Phiếu Lớn (VN30)"],
    )

    limit_stocks = st.slider(
        "Giới hạn số mã quét:",
        min_value=20,
        max_value=400,
        value=100,
        step=20,
        help="Giảm số mã để quét nhanh hơn, tăng số mã để bao quát toàn thị trường",
    )

    min_liq = st.number_input("Thanh khoản tối thiểu (MA20 Volume):", value=50000, step=20000)

    tab_ta, tab_fa = st.tabs(["📊 Kỹ Thuật (TA)", "🏢 Cơ Bản (FA)"])

    with tab_ta:
        req_trend_cb = st.checkbox("Bắt buộc Trend Template (MA50 > 150 > 200)", value=st.session_state.req_trend)
        c_t1, c_t2 = st.columns(2)
        with c_t1:
            min_from_low = st.number_input("% Trên Đáy 52W:", min_value=0, max_value=200, value=20)
        with c_t2:
            max_from_high = st.number_input("% Cách Đỉnh 52W:", min_value=1, max_value=100, value=25)

        req_pivot_cb = st.checkbox("Yêu cầu Phiên Breakout/Pivot", value=st.session_state.req_pivot)
        req_vcp_cb = st.checkbox("Ưu tiên VCP Nén Biến Động / Cạn Vol", value=st.session_state.req_vcp)

        c_p1, c_p2 = st.columns(2)
        with c_p1:
            pivot_price_in = st.number_input("% Tăng giá phiên nổ:", 1.0, 10.0, 2.0, 0.5)
        with c_p2:
            pivot_vol_in = st.number_input("Hệ số Vol nổ:", 1.0, 5.0, 1.3, 0.1)

        vcp_ratio = st.slider("Độ nén ATR5 / ATR20 (VCP):", 0.4, 0.9, 0.8, 0.05)

    with tab_fa:
        use_fa_cb = st.checkbox("Kích hoạt lọc Cơ bản (FA)", value=False)
        min_eps_g = st.slider("Tăng trưởng EPS YoY (%):", 0, 100, 15)
        min_rev_g = st.slider("Tăng trưởng Doanh thu YoY (%):", 0, 100, 10)
        min_roe = st.slider("ROE tối thiểu (%):", 0, 50, 12)

    params = {
        "ma_fast": 50,
        "ma_mid": 150,
        "ma_slow": 200,
        "min_liquidity_vol": min_liq,
        "min_from_low": min_from_low,
        "max_from_high": max_from_high,
        "pivot_price_pct": pivot_price_in,
        "pivot_vol_mult": pivot_vol_in,
        "vcp_threshold": vcp_ratio,
        "use_fa": use_fa_cb,
        "min_eps_g": min_eps_g,
        "min_rev_g": min_rev_g,
        "min_roe": min_roe,
    }

    st.markdown("---")
    btn_start_scan = st.button("🚀 BẮT ĐẦU LỌC CỔ PHIẾU", type="primary", use_container_width=True)


# ==============================================================================
# 7. QUÉT THỊ TRƯỜNG & THU THẬP LOG DEBUG
# ==============================================================================
if "scan_data" not in st.session_state:
    st.session_state.scan_data = None
if "error_logs" not in st.session_state:
    st.session_state.error_logs = []

if btn_start_scan:
    with st.spinner("Đang kết nối danh sách mã niêm yết từ máy chủ..."):
        all_stocks_df = TCBSDataProvider.fetch_all_listed_tickers()

    if exchange_choice == "HOSE (Sàn TP.HCM)":
        target_df = all_stocks_df[all_stocks_df["Exchange"] == "HOSE"]
    elif exchange_choice == "HNX (Sàn Hà Nội)":
        target_df = all_stocks_df[all_stocks_df["Exchange"] == "HNX"]
    elif exchange_choice == "UPCoM":
        target_df = all_stocks_df[all_stocks_df["Exchange"] == "UPCOM"]
    elif exchange_choice == "Top Cổ Phiếu Lớn (VN30)":
        target_df = all_stocks_df[all_stocks_df["Symbol"].isin(TCBSDataProvider.FALLBACK_VN30)]
    else:
        target_df = all_stocks_df

    stock_list = target_df.head(limit_stocks).to_dict("records")
    total_tickers = len(stock_list)

    progress_bar = st.progress(0.0)
    status_box = st.empty()
    status_box.markdown(f"**⚡ Đang phân tích `{total_tickers}` mã cổ phiếu...**")

    success_results = []
    error_list = []
    completed_count = 0

    # Chạy song song đa luồng an toàn
    with ThreadPoolExecutor(max_workers=8) as executor:
        future_map = {
            executor.submit(analyze_stock, row["Symbol"], row["Exchange"], params): row["Symbol"]
            for row in stock_list
        }
        for future in as_completed(future_map):
            completed_count += 1
            res = future.result()
            if res.get("status") == "success":
                success_results.append(res["data"])
            elif res.get("status") == "error":
                error_list.append(f"Lỗi ở mã {res.get('symbol')}: {res.get('error_msg')}")
            
            progress_bar.progress(completed_count / total_tickers)

    progress_bar.empty()
    status_box.empty()
    st.session_state.scan_data = success_results
    st.session_state.error_logs = error_list


# ==============================================================================
# 8. HIỂN THỊ KẾT QUẢ & CONSOLE DEBUG
# ==============================================================================
# Hiển thị console debug nếu có lỗi phát sinh
if st.session_state.error_logs:
    with st.expander("🛠️ Nhật ký lỗi phát sinh (Debug Console)", expanded=False):
        for err in st.session_state.error_logs:
            st.error(err)

if st.session_state.scan_data is not None:
    raw_results = st.session_state.scan_data
    df_res = pd.DataFrame(raw_results)

    if not df_res.empty:
        # Áp dụng các điều kiện lọc bổ sung từ Sidebar
        if req_trend_cb:
            df_res = df_res[df_res["Trend"] == True]
        if req_pivot_cb:
            df_res = df_res[df_res["Pivot"] == True]
        if req_vcp_cb:
            df_res = df_res[df_res["VCP"] == True]
        if use_fa_cb:
            df_res = df_res[df_res["Pass_FA"] == True]

        df_res = df_res.sort_values(by=["Score", "Change_%"], ascending=[False, False]).reset_index(drop=True)

        # KHỐI METRIC TỔNG QUAN
        pass_count = len(df_res)
        total_scanned = len(raw_results)
        win_rate = (pass_count / total_scanned * 100) if total_scanned > 0 else 0
        top_symbol = df_res["Symbol"].iloc[0] if pass_count > 0 else "N/A"

        st.markdown(
            f"""
            <div class="kpi-container">
                <div class="kpi-card">
                    <div class="kpi-label">Tổng mã tải thành công</div>
                    <div class="kpi-value">{total_scanned}</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Mã đạt tiêu chí</div>
                    <div class="kpi-value" style="color: #0ecb81;">{pass_count}</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Tỷ lệ hội tụ</div>
                    <div class="kpi-value">{win_rate:.1f}%</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Điểm số cao nhất</div>
                    <div class="kpi-value" style="color: #2962ff;">{top_symbol}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if not df_res.empty:
            view_df = df_res.copy()
            view_df["Trend_Tag"] = view_df["Trend"].apply(lambda x: "🟢 Đạt Trend" if x else "⚪ Không")
            view_df["VCP_Tag"] = view_df["VCP"].apply(lambda x: "🔥 VCP/Cạn Vol" if x else "-")
            view_df["Pivot_Tag"] = view_df["Pivot"].apply(lambda x: "🚀 Pivot Nổ" if x else "-")

            table_cols = [
                "Symbol", "Exchange", "Price", "Change_%", "Volume",
                "Score", "Trend_Tag", "VCP_Tag", "Pivot_Tag",
                "Cách_Đỉnh_%", "Trên_Đáy_%",
            ]
            if use_fa_cb:
                table_cols += ["ROE", "EPS_G", "Rev_G", "Quarter"]

            st.markdown("#### 📋 BẢNG DANH MỤC CỔ PHIẾU ĐẠT TIÊU CHÍ")

            st.dataframe(
                view_df[table_cols],
                column_config={
                    "Symbol": st.column_config.TextColumn("Mã CP"),
                    "Exchange": st.column_config.TextColumn("Sàn"),
                    "Price": st.column_config.NumberColumn("Giá (k VND)", format="%.2f"),
                    "Change_%": st.column_config.NumberColumn("Thay đổi", format="%+0.2f%%"),
                    "Volume": st.column_config.NumberColumn("Khối lượng", format="%d"),
                    "Score": st.column_config.ProgressColumn("Điểm Kỹ Thuật", format="%d/100", min_value=0, max_value=100),
                    "Trend_Tag": st.column_config.TextColumn("Minervini Trend"),
                    "VCP_Tag": st.column_config.TextColumn("Mẫu Hình VCP"),
                    "Pivot_Tag": st.column_config.TextColumn("Điểm Mua"),
                    "Cách_Đỉnh_%": st.column_config.NumberColumn("Cách Đỉnh 52W", format="-%.1f%%"),
                    "Trên_Đáy_%": st.column_config.NumberColumn("Trên Đáy 52W", format="+%.1f%%"),
                },
                use_container_width=True,
                height=320,
            )

            # Xuất file CSV
            csv_buf = io.StringIO()
            view_df[table_cols].to_csv(csv_buf, index=False, encoding="utf-8-sig")
            st.download_button(
                label="📥 Tải Kết Quả Lọc (CSV)",
                data=csv_buf.getvalue(),
                file_name=f"screener_results_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )

            st.markdown("---")
            st.markdown("### 📊 PHÂN TÍCH KỸ THUẬT & TÀI CHÍNH CHI TIẾT")

            c_select, _ = st.columns([2, 5])
            with c_select:
                selected_stock = st.selectbox(
                    "Chọn mã để phân tích biểu đồ:",
                    options=df_res["Symbol"].tolist(),
                )

            if selected_stock:
                stock_kpi = df_res[df_res["Symbol"] == selected_stock].iloc[0]
                k1, k2, k3, k4, k5, k6 = st.columns(6)
                k1.metric("Mã Cổ Phiếu", f"{stock_kpi['Symbol']} ({stock_kpi['Exchange']})")
                k2.metric("Thị Giá", f"{stock_kpi['Price']:.2f}", f"{stock_kpi['Change_%']:+.2f}%")
                k3.metric("Điểm Kỹ Thuật", f"{stock_kpi['Score']}/100")
                k4.metric("ROE (Gần nhất)", f"{stock_kpi['ROE']}%" if use_fa_cb else "N/A")
                k5.metric("Tăng trưởng EPS", f"{stock_kpi['EPS_G']}%" if use_fa_cb else "N/A")
                k6.metric("Tăng trưởng Doanh thu", f"{stock_kpi['Rev_G']}%" if use_fa_cb else "N/A")

                render_tradingview_chart(selected_stock)
        else:
            st.warning("⚠️ Đã nạp dữ liệu thành công nhưng không có mã nào thỏa mãn bộ tiêu chí khắt khe. Hãy thử nới lỏng các thông số ở Sidebar.")
    else:
        st.error("❌ Không thể nạp dữ liệu từ máy chủ. Bạn hãy mở rộng mục '🛠️ Nhật ký lỗi phát sinh (Debug Console)' ở trên để xem chi tiết mã lỗi.")
else:
    st.info("💡 Bấm nút **'🚀 BẮT ĐẦU LỌC CỔ PHIẾU'** tại thanh bên trái để quét dữ liệu.")