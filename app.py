"""
================================================================================
MINERVINI & LIVERMORE VCP QUANTITATIVE TERMINAL (VIETNAM STOCK MARKET)
--------------------------------------------------------------------------------
- Framework: Streamlit, Plotly, Pandas, SQLite3
- Phương pháp: Mark Minervini (Trend Template & VCP) + Jesse Livermore (Pivot & Risk)
================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import sqlite3
import datetime
import time
from typing import Optional, Dict, Any, List, Tuple

# ==============================================================================
# 0. CẤU HÌNH STREAMLIT PAGE
# ==============================================================================
st.set_page_config(
    page_title="Minervini VCP Quant Terminal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Áp dụng CSS tùy biến cho giao diện Terminal chuyên nghiệp
st.markdown("""
    <style>
        .metric-card {
            background-color: #1e222d;
            border: 1px solid #2a2e39;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 10px;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 15px;
        }
        .stTabs [data-baseweb="tab"] {
            font-weight: 600;
            padding: 8px 16px;
        }
    </style>
""", unsafe_allow_html=True)


# ==============================================================================
# 1. MODULE KẾT NỐI DỮ LIỆU THỰC TẾ (DATA CONNECTOR)
# ==============================================================================
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_stock_ohlcv(symbol: str, days: int = 380) -> Optional[pd.DataFrame]:
    """
    Tải dữ liệu lịch sử nến ngày (OHLCV) từ nguồn dữ liệu chứng khoán VN (TCBS REST Endpoint).
- Cache 30 phút (@st.cache_data) để chống nghẽn và tối ưu tốc độ quét.
- Đảm bảo tối thiểu 250 phiên nến giao dịch phục vụ tính toán MA200 & Đáy 52 tuần.
    """
    symbol = symbol.strip().upper()
    end_time = int(time.time())
    start_time = int(end_time - (days * 86400))
    url = f"https://apipubaws.tcbs.com.vn/stock-insight/v1/stock/bars-long-term?ticker={symbol}&type=stock&resolution=D&from={start_time}&to={end_time}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return None

        raw_data = response.json()
        if not raw_data or "data" not in raw_data or len(raw_data["data"]) == 0:
            return None

        df = pd.DataFrame(raw_data["data"])
        df.rename(columns={
            "tradingDate": "date",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume"
        }, inplace=True)

        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)

        # Loại bỏ các phiên dữ liệu rác/lỗi volume = 0
        df = df[df['volume'] > 0].reset_index(drop=True)
        return df
    except Exception as e:
        return None


# ==============================================================================
# 2. THUẬT TOÁN ĐỊNH LƯỢNG VCP & TREND TEMPLATE (MINERVINI ENGINE)
# ==============================================================================
def check_trend_template(df: pd.DataFrame) -> Tuple[bool, Dict[str, float]]:
    """
    Kiểm tra 5 tiêu chí cốt lõi trong Mark Minervini Trend Template:
1. Giá > MA50 > MA150 > MA200
2. MA200 hướng lên (so với 22 phiên trước đó - tương đương 1 tháng)
3. Giá >= Đáy 52 tuần + 25%
4. Giá >= Đỉnh 52 tuần - 20%
    """
    if len(df) < 220:
        return False, {}

    current_price = df['close'].iloc[-1]
    ma50 = df['MA50'].iloc[-1]
    ma150 = df['MA150'].iloc[-1]
    ma200 = df['MA200'].iloc[-1]
    ma200_1m_ago = df['MA200'].iloc[-22]

    period_52w = min(252, len(df))
    high_52w = df['high'].iloc[-period_52w:].max()
    low_52w = df['low'].iloc[-period_52w:].min()

    c1 = (current_price > ma50) and (ma50 > ma150) and (ma150 > ma200)
    c2 = ma200 > ma200_1m_ago
    c3 = current_price >= (low_52w * 1.25)
    c4 = current_price >= (high_52w * 0.80)

    is_uptrend = c1 and c2 and c3 and c4
    metrics = {
        "current_price": current_price,
        "ma50": ma50,
        "ma150": ma150,
        "ma200": ma200,
        "high_52w": high_52w,
        "low_52w": low_52w,
        "pct_above_52w_low": ((current_price - low_52w) / low_52w) * 100,
        "pct_below_52w_high": ((current_price - high_52w) / high_52w) * 100
    }
    return is_uptrend, metrics


def detect_vcp_pattern(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Thuật toán phát hiện sự thu hẹp biến động (VCP):
- Phân tích nền tích lũy trong 50 phiên gần nhất thành 3 nhịp sóng điều chỉnh (Contractions).
- Kiểm tra tính chất thu hẹp: Biên độ nhịp 1 > Nhịp 2 > Nhịp 3 (ví dụ: -15% -> -8% -> -3%).
- Kiểm tra Volume Dry-Up (VDU): Khối lượng 3-5 phiên gần nhất giảm 40-60% so với MA20.
- Xác định điểm Breakout Pivot & Điểm Cắt lỗ (Hard Stop max 7%).
    """
    res = {
        "is_vcp": False,
        "contractions": [],
        "is_vdu": False,
        "is_breakout": False,
        "pivot_point": 0.0,
        "stop_loss": 0.0,
        "target_price": 0.0,
        "risk_reward": 0.0,
        "vdu_ratio": 0.0
    }

    if len(df) < 60:
        return res

    # 1. Khảo sát 3 nhịp điều chỉnh trong 55 phiên gần nhất
    # Phân đoạn 3 nhịp: Nhịp 1 (-55 đến -35), Nhịp 2 (-35 đến -15), Nhịp 3 (-15 đến -2)
    w1 = df.iloc[-55:-35]
    w2 = df.iloc[-35:-15]
    w3 = df.iloc[-15:-1]

    d1 = ((w1['high'].max() - w1['low'].min()) / w1['high'].max()) * 100
    d2 = ((w2['high'].max() - w2['low'].min()) / w2['high'].max()) * 100
    d3 = ((w3['high'].max() - w3['low'].min()) / w3['high'].max()) * 100

    res["contractions"] = [round(d1, 1), round(d2, 1), round(d3, 1)]

    # Tiêu chuẩn thu hẹp: d1 > d2 > d3 và nhịp cuối cùng d3 phải siết chặt (< 8%)
    is_contracting = (d1 > d2) and (d2 > d3) and (d3 <= 8.5) and (d1 <= 30.0)

    # 2. Khối lượng cạn kiệt (Volume Dry-Up - VDU)
    vol_ma20 = df['VOL_MA20'].iloc[-1]
    last_3d_vol_mean = df['volume'].iloc[-4:-1].mean()
    vdu_ratio = (last_3d_vol_mean / vol_ma20) if vol_ma20 > 0 else 1.0
    res["vdu_ratio"] = round(vdu_ratio, 2)
    # VDU đạt chuẩn nếu volume siết nền giảm 35% - 60% so với MA20
    is_vdu = vdu_ratio <= 0.65

    # 3. Xác định Điểm Pivot (Mức đỉnh cao nhất của nhịp siết nền gần nhất)
    pivot_point = w3['high'].max()
    current_price = df['close'].iloc[-1]
    current_vol = df['volume'].iloc[-1]

    # Điểm bứt phá: Giá đóng cửa >= Pivot và Volume phiên hiện tại > 1.3x MA20
    is_breakout = (current_price >= pivot_point) and (current_vol >= 1.3 * vol_ma20)

    # 4. Quản trị rủi ro & Tính toán Cắt lỗ / Chốt lời
    # Đáy nhịp siết nền cuối cùng (w3 low) làm hỗ trợ tự nhiên
    support_level = w3['low'].min()
    pct_risk_to_support = ((current_price - support_level) / current_price) * 100

    # Minervini Hard-stop: Không bao giờ để lỗ quá 7%. Lấy min giữa hỗ trợ và 6.5%
    if pct_risk_to_support < 7.0 and pct_risk_to_support > 1.5:
        stop_loss = round(support_level * 0.995, 2)
    else:
        stop_loss = round(current_price * 0.935, 2)

    risk_per_share = current_price - stop_loss
    # Mục tiêu Risk/Reward tối thiểu 1:2.5
    target_price = round(current_price + (risk_per_share * 2.5), 2)
    rr_ratio = round((target_price - current_price) / max(0.01, risk_per_share), 2)

    res["is_vcp"] = is_contracting
    res["is_vdu"] = is_vdu
    res["is_breakout"] = is_breakout
    res["pivot_point"] = pivot_point
    res["stop_loss"] = stop_loss
    res["target_price"] = target_price
    res["risk_reward"] = rr_ratio

    return res


def scan_symbol(symbol: str) -> Optional[Dict[str, Any]]:
    """Xử lý tính toán toàn bộ chỉ số cho một mã cổ phiếu."""
    df = fetch_stock_ohlcv(symbol)
    if df is None or len(df) < 220:
        return None

    # Tính toán các chỉ báo kỹ thuật cơ bản
    df['MA50'] = df['close'].rolling(50).mean()
    df['MA150'] = df['close'].rolling(150).mean()
    df['MA200'] = df['close'].rolling(200).mean()
    df['VOL_MA20'] = df['volume'].rolling(20).mean()

    is_uptrend, trend_metrics = check_trend_template(df)
    vcp_data = detect_vcp_pattern(df)

    return {
        "symbol": symbol,
        "df": df,
        "is_uptrend": is_uptrend,
        "trend_metrics": trend_metrics,
        "vcp_data": vcp_data
    }


# ==============================================================================
# 3. MODULE CƠ SỞ DỮ LIỆU: NHẬT KÝ GIAO DỊCH (SQLITE)
# ==============================================================================
DB_NAME = "trading_journal.db"

def init_db(db_name: str = DB_NAME):
    """Khởi tạo cấu trúc bảng SQLite."""
    conn = sqlite3.connect(db_name)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            entry_date TEXT NOT NULL,
            entry_price REAL NOT NULL,
            quantity INTEGER NOT NULL,
            stop_loss REAL NOT NULL,
            target_price REAL NOT NULL,
            exit_price REAL,
            status TEXT NOT NULL,
            reason TEXT,
            pnl REAL,
            pnl_pct REAL
        )
    ''')
    conn.commit()
    conn.close()

init_db()

def db_add_trade(symbol: str, price: float, qty: int, sl: float, tp: float, reason: str, db_name: str = DB_NAME):
    conn = sqlite3.connect(db_name)
    c = conn.cursor()
    date_str = datetime.date.today().strftime("%Y-%m-%d")
    c.execute('''
        INSERT INTO trades (symbol, entry_date, entry_price, quantity, stop_loss, target_price, status, reason, pnl, pnl_pct)
        VALUES (?, ?, ?, ?, ?, ?, 'OPEN', ?, 0, 0)
    ''', (symbol.upper(), date_str, price, qty, sl, tp, reason))
    conn.commit()
    conn.close()

def db_close_trade(trade_id: int, exit_price: float, db_name: str = DB_NAME):
    conn = sqlite3.connect(db_name)
    c = conn.cursor()
    c.execute("SELECT entry_price, quantity FROM trades WHERE id = ?", (trade_id,))
    row = c.fetchone()
    if row:
        entry_price, qty = row[0], row[1]
        pnl = (exit_price - entry_price) * qty
        pnl_pct = ((exit_price - entry_price) / entry_price) * 100
        c.execute('''
            UPDATE trades 
            SET exit_price = ?, status = 'CLOSED', pnl = ?, pnl_pct = ? 
            WHERE id = ?
        ''', (exit_price, pnl, pnl_pct, trade_id))
        conn.commit()
    conn.close()

def db_get_trades(db_name: str = DB_NAME) -> pd.DataFrame:
    conn = sqlite3.connect(db_name)
    df = pd.read_sql_query("SELECT * FROM trades ORDER BY id DESC", conn)
    conn.close()
    return df


# ==============================================================================
# 4. MODULE ĐỒ THỊ KỸ THUẬT TƯƠNG TÁC (PLOTLY CHART)
# ==============================================================================
def render_minervini_chart(df: pd.DataFrame, symbol: str, pivot_price: float, stop_loss: float):
    """Vẽ biểu đồ nến chuyên nghiệp với MA50/150/200, Điểm Pivot, Stoploss và Volume MA20."""
    # Lấy 120 phiên gần nhất để biểu đồ rõ ràng
    plot_df = df.iloc[-120:].copy()

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.75, 0.25]
    )

    # 1. Đồ thị nến (Candlestick)
    fig.add_trace(
        go.Candlestick(
            x=plot_df['date'],
            open=plot_df['open'],
            high=plot_df['high'],
            low=plot_df['low'],
            close=plot_df['close'],
            name="OHLC",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350"
        ),
        row=1, col=1
    )

    # Các đường MA
    fig.add_trace(go.Scatter(x=plot_df['date'], y=plot_df['MA50'], line=dict(color='#2196F3', width=1.5), name="MA50"), row=1, col=1)
    fig.add_trace(go.Scatter(x=plot_df['date'], y=plot_df['MA150'], line=dict(color='#FF9800', width=1.5), name="MA150"), row=1, col=1)
    fig.add_trace(go.Scatter(x=plot_df['date'], y=plot_df['MA200'], line=dict(color='#E91E63', width=2), name="MA200"), row=1, col=1)

    # Vẽ đường Pivot và Stop-Loss
    if pivot_price > 0:
        fig.add_hline(y=pivot_price, line_dash="dash", line_color="#00E676", annotation_text=f"Pivot: {pivot_price:,.0f}", row=1, col=1)
    if stop_loss > 0:
        fig.add_hline(y=stop_loss, line_dash="dot", line_color="#FF1744", annotation_text=f"Stop: {stop_loss:,.0f}", row=1, col=1)

    # 2. Đồ thị Khối lượng (Volume)
    colors = ['#26a69a' if c >= o else '#ef5350' for c, o in zip(plot_df['close'], plot_df['open'])]
    fig.add_trace(go.Bar(x=plot_df['date'], y=plot_df['volume'], marker_color=colors, name="Volume"), row=2, col=1)
    fig.add_trace(go.Scatter(x=plot_df['date'], y=plot_df['VOL_MA20'], line=dict(color='#BA68C8', width=1.5), name="Vol MA20"), row=2, col=1)

    fig.update_layout(
        title=f"<b>Mô hình VCP & Cấu trúc Xu hướng: {symbol}</b>",
        xaxis_rangeslider_visible=False,
        height=550,
        margin=dict(l=10, r=10, t=40, b=10),
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


# ==============================================================================
# 5. GIAO DIỆN NGƯỜI DÙNG STREAMLIT TERMINAL
# ==============================================================================
st.sidebar.title("⚡ Minervini Quant Hub")
st.sidebar.caption("Hệ thống Lọc VCP & Quản trị Rủi ro Tự động")

# Nhập danh sách cổ phiếu cần quét
default_watch = "FPT, SSI, HPG, MWG, DGC, VHC, PVS, PVD, VCI, MBB, TCB, CTG, ACB, FRT, KDH, GMD"
input_tickers = st.sidebar.text_area("Danh sách cổ phiếu theo dõi:", value=default_watch, height=130)

tab1, tab2, tab3 = st.tabs([
    "🔍 Bộ Lọc VCP Real-time",
    "⚖️ Quản Trị Vị Thế & Vốn",
    "📓 Nhật Ký Giao Dịch (SQLite)"
])

# ------------------------------------------------------------------------------
# TAB 1: BỘ LỌC VCP REAL-TIME
# ------------------------------------------------------------------------------
with tab1:
    st.subheader("Bộ Lọc Mô Hình VCP & Breakout Pivot")
    st.caption("Quét toàn diện Trend Template (Giai đoạn 2), Nhịp co hẹp biến động (Contraction) và Thanh khoản cạn kiệt (VDU).")

    if st.button("🚀 Kích Hoạt Quét Toàn Diện", use_container_width=True):
        tickers = [t.strip().upper() for t in input_tickers.split(",") if t.strip()]
        progress = st.progress(0)
        scanner_results = []
        chart_data_store = {}

        for idx, sym in enumerate(tickers):
            data = scan_symbol(sym)
            if data and data["is_uptrend"]:
                vcp = data["vcp_data"]
                status = "⭐ VCP BREAKOUT" if vcp["is_breakout"] else ("🔥 ĐẠT CHUẨN VCP" if vcp["is_vcp"] else "✅ UPTREND (Trend Template)")
                
                scanner_results.append({
                    "Mã CP": sym,
                    "Trạng Thái": status,
                    "Giá Hiện Tại": data["trend_metrics"]["current_price"],
                    "Điểm Pivot": vcp["pivot_point"],
                    "Biên Độ Co Hẹp (%)": str(vcp["contractions"]),
                    "Tỷ Lệ Vol Cạn (VDU)": f"{vcp['vdu_ratio']}x MA20",
                    "Cắt Lỗ Đề Xuất": vcp["stop_loss"],
                    "Chốt Lời (1:2.5)": vcp["target_price"],
                    "Vượt Đáy 52W": f"+{data['trend_metrics']['pct_above_52w_low']:.1f}%",
                    "Cách Đỉnh 52W": f"{data['trend_metrics']['pct_below_52w_high']:.1f}%"
                })
                chart_data_store[sym] = (data["df"], vcp["pivot_point"], vcp["stop_loss"])
            
            progress.progress((idx + 1) / len(tickers))

        if scanner_results:
            st.session_state["scan_results"] = scanner_results
            st.session_state["chart_store"] = chart_data_store
            st.success(f"Hoàn thành! Phát hiện {len(scanner_results)} mã thỏa mãn Trend Template.")
        else:
            st.warning("Không có cổ phiếu nào thỏa mãn các điều kiện lọc.")

    # Hiển thị bảng kết quả & đồ thị kỹ thuật tương tác
    if "scan_results" in st.session_state and st.session_state["scan_results"]:
        res_df = pd.DataFrame(st.session_state["scan_results"])
        st.dataframe(res_df, use_container_width=True)

        st.markdown("---")
        st.subheader("📊 Phân Tích Kỹ Thuật Đa Khung Thời Gian")
        selected_stock = st.selectbox("Chọn mã để hiển thị biểu đồ chi tiết:", list(st.session_state["chart_store"].keys()))
        if selected_stock:
            stock_df, pivot_val, sl_val = st.session_state["chart_store"][selected_stock]
            fig = render_minervini_chart(stock_df, selected_stock, pivot_val, sl_val)
            st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 2: QUẢN TRỊ VỊ THẾ & PHÂN BỔ VỐN (POSITION SIZING)
# ------------------------------------------------------------------------------
with tab2:
    st.subheader("Máy Tính Phân Bổ Vị Thế & Quản Trị Rủi Ro Cố Định")
    st.markdown("""
    > **Quy tắc Vàng Mark Minervini & Jesse Livermore:** 
    > *Không bao giờ mạo hiểm quá 1.0% - 2.0% tổng tài sản cho một giao dịch. Mức cắt lỗ tối đa tuyệt đối là 7%-8%.*
    """)

    col_nav1, col_nav2, col_nav3 = st.columns(3)
    with col_nav1:
        account_nav = st.number_input("Tổng quy mô vốn danh mục (NAV) VNĐ:", min_value=10_000_000, value=500_000_000, step=50_000_000, format="%d")
    with col_nav2:
        risk_tolerance_pct = st.slider("Tỷ lệ rủi ro tối đa chấp nhận (% NAV):", min_value=0.5, max_value=2.5, value=1.5, step=0.1)
    with col_nav3:
        max_monetary_risk = account_nav * (risk_tolerance_pct / 100)
        st.metric("Rủi ro tối đa cho phép / 1 lệnh", f"{max_monetary_risk:,.0f} VNĐ")

    st.markdown("---")
    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        entry_p = st.number_input("Giá mua kế hoạch (VNĐ):", value=45000, step=500)
    with col_p2:
        stop_p = st.number_input("Giá cắt lỗ kế hoạch (VNĐ):", value=42000, step=500)
    with col_p3:
        target_p = st.number_input("Giá mục tiêu chốt lời (VNĐ):", value=52500, step=500)

    # Tính toán định lượng vị thế
    if entry_p > stop_p:
        loss_per_share = entry_p - stop_p
        loss_pct = (loss_per_share / entry_p) * 100
        reward_per_share = target_p - entry_p
        rr_calc = (reward_per_share / loss_per_share) if loss_per_share > 0 else 0

        # Số lượng CP tối đa được mua để không vượt quá mức rủi ro NAV
        calc_shares = int(max_monetary_risk // loss_per_share)
        total_trade_val = calc_shares * entry_p
        portfolio_weight = (total_trade_val / account_nav) * 100

        st.markdown("### Kết Quả Định Lượng Phân Bổ:")
        c_res1, c_res2, c_res3, c_res4 = st.columns(4)
        c_res1.metric("Khối lượng mua tối đa", f"{calc_shares:,.0f} CP")
        c_res2.metric("Tỷ trọng giải ngân / NAV", f"{portfolio_weight:.1f}%")
        c_res3.metric("Tỷ lệ Cắt Lỗ", f"-{loss_pct:.2f}%", delta=f"-{loss_pct:.2f}%", delta_color="inverse")
        c_res4.metric("Tỷ lệ Risk : Reward", f"1 : {rr_calc:.2f}")

        # Cảnh báo vi phạm nguyên tắc Minervini
        if loss_pct > 7.0:
            st.error(f"⚠️ **CẢNH BÁO:** Mức cắt lỗ ({loss_pct:.2f}%) vượt quá ngưỡng an toàn tối đa 7% của Mark Minervini!")
        elif rr_calc < 2.0:
            st.warning(f"⚠️ **LƯU Ý:** Tỷ lệ Risk/Reward (1:{rr_calc:.2f}) nhỏ hơn 1:2.0 tiêu chuẩn.")
        else:
            st.success("✅ Vị thế đạt chuẩn cả về Quản trị rủi ro và Tỷ lệ sinh lời/rủi ro!")

# ------------------------------------------------------------------------------
# TAB 3: NHẬT KÝ GIAO DỊCH (SQLITE)
# ------------------------------------------------------------------------------
with tab3:
    st.subheader("Nhật Ký Giao Dịch & Hiệu Suất Hệ Thống (Quantitative Journal)")

    with st.expander("➕ Ghi nhận vị thế vào lệnh mới", expanded=False):
        c_in1, c_in2, c_in3, c_in4 = st.columns(4)
        with c_in1:
            j_sym = st.text_input("Mã cổ phiếu:", "FPT").upper()
            j_entry = st.number_input("Giá mua:", value=100000, step=1000)
        with c_in2:
            j_qty = st.number_input("Số lượng CP:", value=1000, step=100)
            j_sl = st.number_input("Cắt lỗ:", value=93500, step=500)
        with c_in3:
            j_tp = st.number_input("Chốt lời:", value=116000, step=1000)
            j_reason = st.text_input("Lý do vào lệnh:", "Minervini VCP 3T Breakout")
        with c_in4:
            st.write("")
            st.write("")
            if st.button("Lưu Vị Thế", use_container_width=True):
                if j_entry > j_sl:
                    db_add_trade(j_sym, j_entry, j_qty, j_sl, j_tp, j_reason)
                    st.success(f"Đã lưu thành công lệnh mua {j_sym}!")
                    st.rerun()
                else:
                    st.error("Giá cắt lỗ phải nhỏ hơn giá vào lệnh!")

    # Lấy danh sách lệnh
    trades_df = db_get_trades()
    open_pos = trades_df[trades_df['status'] == 'OPEN'] if not trades_df.empty else pd.DataFrame()
    closed_pos = trades_df[trades_df['status'] == 'CLOSED'] if not trades_df.empty else pd.DataFrame()

    # Thống kê hiệu suất nếu có lệnh đã đóng
    if not closed_pos.empty:
        total_closed = len(closed_pos)
        wins = closed_pos[closed_pos['pnl'] > 0]
        losses = closed_pos[closed_pos['pnl'] <= 0]

        win_rate = (len(wins) / total_closed) * 100
        total_pnl = closed_pos['pnl'].sum()
        gross_profit = wins['pnl'].sum()
        gross_loss = abs(losses['pnl'].sum())
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 999.0

        col_st1, col_st2, col_st3, col_st4 = st.columns(4)
        col_st1.metric("Tổng Số Lệnh Đã Đóng", f"{total_closed}")
        col_st2.metric("Tỷ Lệ Thắng (Win-Rate)", f"{win_rate:.1f}%")
        col_st3.metric("Profit Factor", f"{profit_factor:.2f}")
        col_st4.metric("Tổng P&L Thực Tế", f"{total_pnl:,.0f} VNĐ", delta=f"{total_pnl:,.0f}")

    st.markdown("---")
    st.subheader("Vị Thế Đang Mở (Open Trades)")
    if not open_pos.empty:
        st.dataframe(open_pos[['id', 'symbol', 'entry_date', 'entry_price', 'quantity', 'stop_loss', 'target_price', 'reason']], use_container_width=True)

        col_cl1, col_cl2, col_cl3 = st.columns([2, 2, 2])
        with col_cl1:
            sel_id = st.selectbox("Chọn ID lệnh cần đóng:", open_pos['id'].tolist())
        with col_cl2:
            real_exit = st.number_input("Giá chốt/thoát thực tế (VNĐ):", value=float(open_pos.loc[open_pos['id'] == sel_id, 'entry_price'].values[0]), step=500.0)
        with col_cl3:
            st.write("")
            if st.button("🔒 Đóng Vị Thế Ngay", use_container_width=True):
                db_close_trade(sel_id, real_exit)
                st.success(f"Đã đóng thành công lệnh #{sel_id}!")
                st.rerun()
    else:
        st.info("Hiện không có vị thế nào đang mở.")

    st.markdown("---")
    st.subheader("Lịch Sử Lệnh Đã Hoàn Thành (Closed Trades)")
    if not closed_pos.empty:
        st.dataframe(closed_pos[['id', 'symbol', 'entry_date', 'entry_price', 'exit_price', 'quantity', 'pnl', 'pnl_pct', 'reason']], use_container_width=True)
    else:
        st.caption("Chưa có lịch sử lệnh đã đóng.")