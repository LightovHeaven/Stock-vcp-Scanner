"""
================================================================================
DỰ ÁN: VN-STOCK SCREENER PRO (TRADINGVIEW & FIREAT STYLE)
PHIÊN BẢN: 2.5 (ENTERPRISE EDITION)
CHỨC NĂNG: Lọc cổ phiếu toàn thị trường (HOSE, HNX, UPCoM) theo SEPA & Pivot
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
# 1. THIẾT LẬP CẤU HÌNH TRANG VÀ GIAO DIỆN DARK PRO (TRADINGVIEW STYLE)
# ==============================================================================
st.set_page_config(
    page_title="VN-Stock Screener Pro | Institutional Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS tạo giao diện bảng điện tử Dark Slate chuyên nghiệp
st.markdown(
    """
    <style>
    /* Tổng quan nền tối */
    .stApp {
        background-color: #0b0e14;
        color: #d1d4dc;
        font-family: -apple-system, BlinkMacSystemFont, 'Trebuchet MS', Roboto, Ubuntu, sans-serif;
    }
    
    /* Header Terminal */
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
        letter-spacing: 0.5px;
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

    /* KPI Summary Cards */
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
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    .kpi-label {
        font-size: 12px;
        color: #787b86;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .kpi-value {
        font-size: 22px;
        font-weight: 700;
        color: #f0f3fa;
        margin-top: 4px;
    }

    /* Tùy biến Sidebar */
    [data-testid="stSidebar"] {
        background-color: #131722;
        border-right: 1px solid #2a2e39;
    }
    
    /* Preset buttons */
    div.stButton > button {
        background-color: #2a2e39;
        color: #d1d4dc;
        border: 1px solid #363a45;
        border-radius: 6px;
        font-weight: 600;
        transition: all 0.2s ease;
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
# 2. DATA PROVIDER: TRUY XUẤT VÀ CACHE DỮ LIỆU TỪ MÁY CHỦ TCBS
# ==============================================================================
class TCBSDataProvider:
    """Xử lý kết nối và bóc tách dữ liệu từ API máy chủ công khai của TCBS."""

    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json",
    }

    # Danh sách dự phòng khi mạng gặp sự cố
    FALLBACK_VN30 = [
        "ACB", "BCM", "BID", "BVH", "CTG", "FPT", "GAS", "GVR", "HDB", "HPG",
        "MBB", "MSN", "MWG", "PLX", "POW", "SAB", "SHB", "SSB", "SSI", "STB",
        "TCB", "TPB", "VCB", "VHM", "VIB", "VIC", "VJC", "VNM", "VPB", "VRE",
    ]

    @staticmethod
    @st.cache_data(ttl=86400, show_spinner=False)
    def fetch_all_listed_tickers() -> pd.DataFrame:
        """Lấy toàn bộ danh sách mã cổ phiếu kèm sàn giao dịch (HOSE, HNX, UPCOM)."""
        endpoints = [
            "https://apipubaws.tcbs.com.vn/tcanalysis/v1/finance/all-stock-symbol",
            "https://apipubaws.tcbs.com.vn/stock-insight/v1/stock/stock-symbol",
        ]

        for url in endpoints:
            try:
                res = requests.get(url, headers=TCBSDataProvider.HEADERS, timeout=8)
                if res.status_code == 200:
                    data = res.json()
                    # Chuẩn hóa cấu trúc trả về
                    symbols_raw = data if isinstance(data, list) else data.get("data", [])
                    if symbols_raw:
                        rows = []
                        for item in symbols_raw:
                            if isinstance(item, dict):
                                code = item.get("ticker") or item.get("symbol") or item.get("code")
                                floor = item.get("floor") or item.get("exchange") or "HOSE"
                                stype = item.get("type", "stock")
                            elif isinstance(item, str):
                                code, floor, stype = item, "HOSE", "stock"
                            else:
                                continue

                            # Chỉ lấy cổ phiếu phổ thông (độ dài 3 ký tự, bỏ qua chứng quyền/trái phiếu)
                            if code and len(code) == 3 and stype in ("stock", "STOCK", ""):
                                rows.append({"Symbol": code.upper(), "Exchange": floor.upper()})

                        if rows:
                            df_tickers = pd.DataFrame(rows).drop_duplicates(subset=["Symbol"])
                            return df_tickers
            except Exception:
                continue

        # Fallback danh sách nòng cốt nếu API liệt kê toàn sàn bị gián đoạn
        fb_data = [{"Symbol": s, "Exchange": "HOSE"} for s in TCBSDataProvider.FALLBACK_VN30]
        return pd.DataFrame(fb_data)

    @staticmethod
    @st.cache_data(ttl=300, show_spinner=False)
    def get_price_history(symbol: str, days_back: int = 400) -> pd.DataFrame:
        """Lấy lịch sử giá nến ngày (OHLCV) có cache 5 phút để chống rate-limit."""
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
            if not payload:
                return pd.DataFrame()

            df = pd.DataFrame(payload)
            df["tradingDate"] = pd.to_datetime(df["tradingDate"])
            df = df.sort_values("tradingDate").reset_index(drop=True)

            df = df[["tradingDate", "open", "high", "low", "close", "volume"]]
            df.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]

            # Đồng bộ đơn vị giá về nghìn đồng (k VND)
            if df["Close"].iloc[-1] > 1000:
                df[["Open", "High", "Low", "Close"]] = df[["Open", "High", "Low", "Close"]] / 1000.0

            return df
        except Exception:
            return pd.DataFrame()

    @staticmethod
    @st.cache_data(ttl=3600, show_spinner=False)
    def get_financial_ratios(symbol: str) -> Dict[str, float]:
        """Truy xuất chỉ số tài chính cơ bản theo quý (ROE, EPS Growth, Doanh thu Growth)."""
        fallback_res = {
            "roe": 0.0,
            "eps_growth_yoy": 0.0,
            "rev_growth_yoy": 0.0,
            "quarter": "N/A",
        }
        try:
            url = f"https://apipubaws.tcbs.com.vn/tcanalysis/v1/finance/{symbol}/financialratio?yearly=0&isAll=false"
            res = requests.get(url, headers=TCBSDataProvider.HEADERS, timeout=5)
            if res.status_code != 200:
                return fallback_res

            items = res.json()
            if not isinstance(items, list) or len(items) == 0:
                return fallback_res

            # Sắp xếp quý gần nhất lên đầu
            items = sorted(
                items,
                key=lambda x: (x.get("year", 0), x.get("quarter", 0)),
                reverse=True,
            )
            latest = items[0]

            # Tính toán ROE
            roe_val = float(latest.get("roe", 0) or 0)
            if 0 < roe_val <= 1.0:
                roe_val *= 100

            # Tính tăng trưởng so với cùng kỳ năm ngoái (4 quý trước)
            eps_g, rev_g = 0.0, 0.0
            if len(items) >= 5:
                yoy_item = items[4]
                curr_eps = float(latest.get("earningPerShare", 0) or 0)
                past_eps = float(yoy_item.get("earningPerShare", 0) or 0)
                if past_eps != 0:
                    eps_g = ((curr_eps - past_eps) / abs(past_eps)) * 100

                curr_rev = float(latest.get("netRevenue", 0) or latest.get("revenue", 0) or 0)
                past_rev = float(yoy_item.get("netRevenue", 0) or yoy_item.get("revenue", 0) or 0)
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
# 3. CORE ENGINE: PHÂN TÍCH KỸ THUẬT MINERVINI SEPA & LIVERMORE PIVOT
# ==============================================================================
def screen_stock(symbol: str, exchange: str, params: dict) -> Optional[Dict]:
    """Phân tích 1 mã đơn lẻ theo toàn bộ tiêu chí Kỹ thuật & Cơ bản."""
    try:
        df = TCBSDataProvider.get_price_history(symbol, days_back=380)
        # Bắt buộc có đủ 210 phiên để tính MA200 và kiểm tra xu hướng 20 phiên trước
        if df.empty or len(df) < 210:
            return None

        # 1. Tính toán các đường Trung bình Động (MA) & Khối lượng
        df["MA50"] = df["Close"].rolling(window=params["ma_fast"]).mean()
        df["MA150"] = df["Close"].rolling(window=params["ma_mid"]).mean()
        df["MA200"] = df["Close"].rolling(window=params["ma_slow"]).mean()
        df["MA20_Vol"] = df["Volume"].rolling(window=20).mean()

        curr = df.iloc[-1]
        prev = df.iloc[-2]
        ma200_20d_ago = df["MA200"].iloc[-21]

        close_p = float(curr["Close"])
        prev_p = float(prev["Close"])
        change_pct = ((close_p - prev_p) / prev_p) * 100
        vol_curr = float(curr["Volume"])
        ma20_vol = float(curr["MA20_Vol"]) if curr["MA20_Vol"] > 0 else 1.0

        # Kiểm tra ngưỡng thanh khoản tối thiểu
        if ma20_vol < params["min_liquidity_vol"]:
            return None

        # 2. Đỉnh/Đáy 52 tuần
        lookback_len = min(len(df), 250)
        high_52w = float(df["High"].iloc[-lookback_len:].max())
        low_52w = float(df["Low"].iloc[-lookback_len:].min())

        pct_above_low = ((close_p - low_52w) / low_52w) * 100
        pct_below_high = ((high_52w - close_p) / high_52w) * 100

        # 3. Tiêu chí Trend Template (Mark Minervini SEPA)
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
        atr5 = float(df["TR"].rolling(5).mean().iloc[-1])
        atr20 = float(df["TR"].rolling(20).mean().iloc[-1])

        is_vcp = atr5 < (atr20 * params["vcp_threshold"])
        vol_3d_avg = float(df["Volume"].iloc[-3:].mean())
        is_vol_dryup = vol_3d_avg < (0.8 * ma20_vol)

        # 5. Điểm nổ Livermore Pivot Breakout
        is_pivot = (change_pct >= params["pivot_price_pct"]) and (vol_curr >= (params["pivot_vol_mult"] * ma20_vol))

        # 6. Dữ liệu Cơ bản (FA)
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

        # 7. Chấm điểm kỹ thuật (Thang điểm 100)
        score = 0
        if c_ma_stack: score += 30
        if c_ma200_up: score += 15
        if c_above_low and c_near_high: score += 15
        if is_vcp or is_vol_dryup: score += 15
        if is_pivot: score += 25

        return {
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
    except Exception:
        return None


# ==============================================================================
# 4. PLOTLY CHART ENGINE (GIAO DIỆN NỀN TỐI CHUYÊN NGHIỆP)
# ==============================================================================
def render_tradingview_chart(symbol: str):
    """Vẽ biểu đồ Nến Nhật, Khối lượng và 3 đường MA theo phong cách TradingView."""
    df = TCBSDataProvider.get_price_history(symbol, days_back=260)
    if df.empty:
        st.warning(f"Không thể tải đồ thị kỹ thuật cho mã {symbol}.")
        return

    df["MA50"] = df["Close"].rolling(50).mean()
    df["MA150"] = df["Close"].rolling(150).mean()
    df["MA200"] = df["Close"].rolling(200).mean()
    df["MA20_Vol"] = df["Volume"].rolling(20).mean()

    # Màu nến theo chuẩn quốc tế: Tăng (Xanh ngọc #089981), Giảm (Đỏ hồng #f23645)
    candle_colors = ["#089981" if c >= o else "#f23645" for o, c in zip(df["Open"], df["Close"])]

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.72, 0.28],
        specs=[[{"type": "candlestick"}], [{"type": "bar"}]],
    )

    # Nến Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df["Date"],
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Giá",
            increasing_line_color="#089981",
            decreasing_line_color="#f23645",
        ),
        row=1,
        col=1,
    )

    # Đường MA
    fig.add_trace(go.Scatter(x=df["Date"], y=df["MA50"], line=dict(color="#2962ff", width=1.5), name="MA50 (Ngắn hạn)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["MA150"], line=dict(color="#ff9800", width=1.5), name="MA150 (Trung hạn)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["MA200"], line=dict(color="#e91e63", width=2), name="MA200 (Dài hạn)"), row=1, col=1)

    # Khối lượng giao dịch
    fig.add_trace(go.Bar(x=df["Date"], y=df["Volume"], marker_color=candle_colors, name="Volume"), row=2, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["MA20_Vol"], line=dict(color="#ffeb3b", width=1.2), name="MA20 Vol"), row=2, col=1)

    # Cấu hình giao diện biểu đồ
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
# 5. HEADER SECTION (TRADING TERMINAL HEADER)
# ==============================================================================
st.markdown(
    """
    <div class="header-box">
        <div>
            <div class="header-title">⚡ VN-STOCK SCREENER PRO</div>
            <div class="header-desc">Hệ thống Lọc Cổ Phiếu Tăng Trưởng Toàn Thị Trường • Minervini SEPA & Livermore Pocket Pivot</div>
        </div>
        <div class="badge-live">HỆ THỐNG KẾT NỐI REALTIME</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# 6. SIDEBAR CẤU HÌNH BỘ LỌC CHUYÊN NGHIỆP (TABS & PRESETS)
# ==============================================================================
# Khởi tạo session state lưu tham số lọc nếu chưa có
if "preset" not in st.session_state:
    st.session_state.preset = "custom"
if "req_trend" not in st.session_state:
    st.session_state.req_trend = True
if "req_pivot" not in st.session_state:
    st.session_state.req_pivot = False
if "req_vcp" not in st.session_state:
    st.session_state.req_vcp = False
if "pivot_price" not in st.session_state:
    st.session_state.pivot_price = 2.5
if "pivot_vol" not in st.session_state:
    st.session_state.pivot_vol = 1.4

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
            st.session_state.pivot_price = 3.0
            st.session_state.pivot_vol = 1.5
            st.rerun()

    st.markdown("---")

    # CHỌN KHÔNG GIAN CỔ PHIẾU (UNIVERSE SELECTION)
    st.markdown("<small style='color:#787b86;'>PHẠM VI THỊ TRƯỜNG:</small>", unsafe_allow_html=True)
    exchange_choice = st.selectbox(
        "Sàn Giao Dịch:",
        ["Tất cả (HOSE, HNX, UPCoM)", "HOSE (Sàn TP.HCM)", "HNX (Sàn Hà Nội)", "UPCoM", "Top Cổ Phiếu Lớn (VN30)"],
    )

    limit_stocks = st.slider(
        "Giới hạn số mã quét (để tối ưu tốc độ):",
        min_value=30,
        max_value=600,
        value=150,
        step=30,
        help="Giảm số mã để quét siêu tốc, tăng số mã để quét sâu toàn thị trường",
    )

    min_liq = st.number_input(
        "Thanh khoản tối thiểu (Khối lượng TB 20 phiên):",
        value=100000,
        step=50000,
        format="%d",
    )

    # TABS CHI TIẾT
    tab_ta, tab_fa = st.tabs(["📊 Kỹ Thuật (TA)", "🏢 Cơ Bản (FA)"])

    with tab_ta:
        st.markdown("**Trend Template (Minervini)**")
        req_trend_cb = st.checkbox("Bắt buộc MA50 > 150 > 200 & MA200 dốc lên", value=st.session_state.req_trend)
        
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            min_from_low = st.number_input("% Trên Đáy 52W:", min_value=0, max_value=200, value=25)
        with col_t2:
            max_from_high = st.number_input("% Cách Đỉnh 52W:", min_value=1, max_value=100, value=25)

        st.markdown("**Điểm Nổ & Mẫu Hình (Pivot/VCP)**")
        req_pivot_cb = st.checkbox("Yêu cầu Phiên bùng nổ (Breakout)", value=st.session_state.req_pivot)
        req_vcp_cb = st.checkbox("Ưu tiên VCP Nén Biến Động / Cạn Vol", value=st.session_state.req_vcp)

        col_p1, col_p2 = st.columns(2)
        with col_p1:
            pivot_price_in = st.number_input("% Tăng giá phiên nổ:", 1.0, 10.0, float(st.session_state.pivot_price), 0.5)
        with col_p2:
            pivot_vol_in = st.number_input("Hệ số Vol bùng nổ:", 1.0, 5.0, float(st.session_state.pivot_vol), 0.2)

        vcp_ratio = st.slider("Độ nén ATR5 / ATR20 (VCP):", 0.4, 0.9, 0.75, 0.05)

    with tab_fa:
        st.markdown("**Chỉ Số Tài Chính Doanh Nghiệp (Quý)**")
        use_fa_cb = st.checkbox("Kích hoạt bộ lọc Cơ bản (FA)", value=False)
        min_eps_g = st.slider("Tăng trưởng EPS YoY tối thiểu (%):", 0, 100, 20)
        min_rev_g = st.slider("Tăng trưởng Doanh thu YoY (%):", 0, 100, 15)
        min_roe = st.slider("ROE tối thiểu (%):", 0, 50, 15)

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
# 7. MAIN ENGINE: TIẾN HÀNH QUÉT TOÀN THỊ TRƯỜNG ĐA LUỒNG
# ==============================================================================
if "scan_data" not in st.session_state:
    st.session_state.scan_data = None

if btn_start_scan:
    with st.spinner("Đang đồng bộ danh mục mã niêm yết từ các sàn giao dịch..."):
        all_stocks_df = TCBSDataProvider.fetch_all_listed_tickers()

    # Lọc theo sàn được chọn
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

    # Giới hạn số lượng mã để tối ưu trải nghiệm người dùng
    stock_list = target_df.head(limit_stocks).to_dict("records")
    total_tickers = len(stock_list)

    progress_bar = st.progress(0.0)
    status_box = st.empty()
    status_box.markdown(f"**⚡ Đang quét song song `{total_tickers}` mã cổ phiếu...**")

    results = []
    completed_count = 0

    # Sử dụng ThreadPoolExecutor xử lý tối đa 12 luồng cùng lúc
    with ThreadPoolExecutor(max_workers=12) as executor:
        future_map = {
            executor.submit(screen_stock, row["Symbol"], row["Exchange"], params): row["Symbol"]
            for row in stock_list
        }
        for future in as_completed(future_map):
            completed_count += 1
            res = future.result()
            if res:
                results.append(res)
            progress_bar.progress(completed_count / total_tickers)

    progress_bar.empty()
    status_box.empty()
    st.session_state.scan_data = results


# ==============================================================================
# 8. WORKSPACE HIỂN THỊ KẾT QUẢ & PHÂN TÍCH CHUYÊN SÂU
# ==============================================================================
if st.session_state.scan_data is not None:
    raw_results = st.session_state.scan_data
    df_res = pd.DataFrame(raw_results)

    if not df_res.empty:
        # Áp dụng các bộ lọc checkbox người dùng chọn
        if req_trend_cb:
            df_res = df_res[df_res["Trend"] == True]
        if req_pivot_cb:
            df_res = df_res[df_res["Pivot"] == True]
        if req_vcp_cb:
            df_res = df_res[df_res["VCP"] == True]
        if use_fa_cb:
            df_res = df_res[df_res["Pass_FA"] == True]

        # Sắp xếp theo Điểm Kỹ Thuật (Score) và % Tăng giá trong ngày
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
                    <div class="kpi-label">Tổng mã đã quét</div>
                    <div class="kpi-value">{total_scanned}</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Số mã đạt tiêu chí</div>
                    <div class="kpi-value" style="color: #0ecb81;">{pass_count}</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Tỷ lệ hội tụ tiêu chuẩn</div>
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
            # Gán nhãn trực quan cho bảng
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

            # Cấu hình hiển thị bảng chuẩn Dataframe Pro
            st.dataframe(
                view_df[table_cols],
                column_config={
                    "Symbol": st.column_config.TextColumn("Mã CP", help="Mã cổ phiếu"),
                    "Exchange": st.column_config.TextColumn("Sàn"),
                    "Price": st.column_config.NumberColumn("Giá (k VND)", format="%.2f"),
                    "Change_%": st.column_config.NumberColumn(
                        "Thay đổi",
                        format="%+0.2f%%",
                        help="Tăng/Giảm so với phiên trước",
                    ),
                    "Volume": st.column_config.NumberColumn("Khối lượng", format="%d"),
                    "Score": st.column_config.ProgressColumn(
                        "Điểm Kỹ Thuật",
                        format="%d/100",
                        min_value=0,
                        max_value=100,
                    ),
                    "Trend_Tag": st.column_config.TextColumn("Minervini Trend"),
                    "VCP_Tag": st.column_config.TextColumn("Mẫu Hình VCP"),
                    "Pivot_Tag": st.column_config.TextColumn("Điểm Mua"),
                    "Cách_Đỉnh_%": st.column_config.NumberColumn("Cách Đỉnh 52W", format="-%.1f%%"),
                    "Trên_Đáy_%": st.column_config.NumberColumn("Trên Đáy 52W", format="+%.1f%%"),
                },
                use_container_width=True,
                height=320,
            )

            # Nút xuất dữ liệu CSV
            csv_buf = io.StringIO()
            view_df[table_cols].to_csv(csv_buf, index=False, encoding="utf-8-sig")
            st.download_button(
                label="📥 Tải Báo Cáo CSV (Excel Ready)",
                data=csv_buf.getvalue(),
                file_name=f"screener_results_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )

            st.markdown("---")

            # INTERACTIVE CHART & KPI SECTION
            st.markdown("### 📊 PHÂN TÍCH KỸ THUẬT & TÀI CHÍNH CHI TIẾT")
            
            c_select, _ = st.columns([2, 5])
            with c_select:
                selected_stock = st.selectbox(
                    "Chọn mã để phân tích biểu đồ chuyên sâu:",
                    options=df_res["Symbol"].tolist(),
                )

            if selected_stock:
                stock_kpi = df_res[df_res["Symbol"] == selected_stock].iloc[0]

                # Bảng Dashboard KPI nhanh cho cổ phiếu được chọn
                k1, k2, k3, k4, k5, k6 = st.columns(6)
                k1.metric("Mã Cổ Phiếu", f"{stock_kpi['Symbol']} ({stock_kpi['Exchange']})")
                k2.metric("Thị Giá", f"{stock_kpi['Price']:.2f}", f"{stock_kpi['Change_%']:+.2f}%")
                k3.metric("Điểm Kỹ Thuật", f"{stock_kpi['Score']}/100")
                k4.metric("ROE (Gần nhất)", f"{stock_kpi['ROE']}%" if use_fa_cb else "N/A")
                k5.metric("Tăng trưởng EPS", f"{stock_kpi['EPS_G']}%" if use_fa_cb else "N/A")
                k6.metric("Tăng trưởng Doanh thu", f"{stock_kpi['Rev_G']}%" if use_fa_cb else "N/A")

                # Vẽ biểu đồ TradingView
                render_tradingview_chart(selected_stock)
        else:
            st.warning("⚠️ Không có cổ phiếu nào thỏa mãn toàn bộ các điều kiện khắt khe đã chọn. Vui lòng nới lỏng các tham số trong Sidebar.")
    else:
        st.error("❌ Không thể nạp dữ liệu. Vui lòng kiểm tra lại kết nối mạng hoặc thử lại sau vài giây.")
else:
    # Màn hình chờ hướng dẫn khi chưa bấm quét
    st.info(
        "💡 **Hướng dẫn bắt đầu:** Thiết lập các thông số chiến lược hoặc chọn **'SEPA Chuẩn'** / **'Pivot Nổ'** "
        "tại thanh bên trái (Sidebar), sau đó bấm **'🚀 BẮT ĐẦU LỌC CỔ PHIẾU'** để hệ thống tiến hành rà soát thị trường."
    )