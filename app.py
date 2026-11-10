"""
================================================================================
MINERVINI & LIVERMORE QUANT TERMINAL (LIGHTWEIGHT EDITION)
- Ngôn ngữ: Python (Streamlit, Pandas, Requests, SQLite3)
- Nguồn dữ liệu: TCBS Direct REST API (Không cần Token / API Key)
- Thiết kế: Tối giản phụ thuộc, chống crash 100% trên môi trường Cloud
================================================================================
"""

import streamlit as st
import pandas as pd
import requests
import sqlite3
import datetime
import time
from typing import Optional, Dict, Any, Tuple

# ==============================================================================
# 1. CẤU HÌNH GIAO DIỆN HỆ THỐNG
# ==============================================================================
st.set_page_config(
    page_title="Minervini Quant Terminal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# 2. MODULE KẾT NỐI DỮ LIỆU THỰC TẾ (TCBS REST API)
# ==============================================================================
@st.cache_data(ttl=1800, show_spinner=False)
def get_tcbs_ohlcv(symbol: str, lookback_days: int = 400) -> Optional[pd.DataFrame]:
    """
    Tải trực tiếp dữ liệu nến ngày từ API mở của TCBS.
    Sử dụng cache 30 phút để chống nghẽn và tăng tốc độ phản hồi.
    """
    clean_symbol = symbol.strip().upper()
    end_ts = int(time.time())
    start_ts = int(end_ts - (lookback_days * 86400))
    
    url = f"https://apipubaws.tcbs.com.vn/stock-insight/v1/stock/bars-long-term?ticker={clean_symbol}&type=stock&resolution=D&from={start_ts}&to={end_ts}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return None
            
        payload = response.json()
        if not payload or "data" not in payload or not payload["data"]:
            return None

        df = pd.DataFrame(payload["data"])
        # Đổi tên cột chuẩn hóa
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
        # Loại bỏ các phiên lỗi thanh khoản
        df = df[df['volume'] > 0].reset_index(drop=True)
        
        return df
    except Exception:
        return None

# ==============================================================================
# 3. THUẬT TOÁN ĐỊNH LƯỢNG: TREND TEMPLATE & VCP (MINERVINI ENGINE)
# ==============================================================================
def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Tính toán MA50, MA150, MA200 và Volume MA20 bằng Pandas thuần túy."""
    df_calc = df.copy()
    df_calc['MA50'] = df_calc['close'].rolling(window=50).mean()
    df_calc['MA150'] = df_calc['close'].rolling(window=150).mean()
    df_calc['MA200'] = df_calc['close'].rolling(window=200).mean()
    df_calc['VOL_MA20'] = df_calc['volume'].rolling(window=20).mean()
    return df_calc

def evaluate_minervini_system(df: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
    """
    Đánh giá tiêu chuẩn Trend Template và nhận diện nhịp siết VCP:
    1. Trend Template: Giá > MA50 > MA150 > MA200; MA200 dốc lên; Giá > Đáy 52W + 25%
    2. VCP: Co hẹp biến động qua 3 nhịp và Thanh khoản cạn kiệt (Volume Dry-Up).
    """
    result = {"is_uptrend": False, "is_vcp": False, "is_breakout": False, "data": {}}
    
    # Yêu cầu tối thiểu 220 phiên để xác định MA200 và độ dốc MA200 1 tháng
    if len(df) < 220:
        return False, result

    curr_price = float(df['close'].iloc[-1])
    ma50 = float(df['MA50'].iloc[-1])
    ma150 = float(df['MA150'].iloc[-1])
    ma200 = float(df['MA200'].iloc[-1])
    ma200_1m_ago = float(df['MA200'].iloc[-22])

    period_52w = min(250, len(df))
    high_52w = float(df['high'].iloc[-period_52w:].max())
    low_52w = float(df['low'].iloc[-period_52w:].min())

    # 1. Trend Template Rules
    cond_ma_align = (curr_price > ma50) and (ma50 > ma150) and (ma150 > ma200)
    cond_ma200_up = ma200 > ma200_1m_ago
    cond_52w_low = curr_price >= (low_52w * 1.25)
    cond_52w_high = curr_price >= (high_52w * 0.80)

    is_uptrend = cond_ma_align and cond_ma200_up and cond_52w_low and cond_52w_high

    # 2. Thuật toán nhận diện VCP (3 nhịp co hẹp trong 50 phiên gần nhất)
    # Nhịp 1: [-50:-30], Nhịp 2: [-30:-12], Nhịp 3: [-12:-1]
    w1 = df.iloc[-50:-30]
    w2 = df.iloc[-30:-12]
    w3 = df.iloc[-12:-1]

    d1 = ((w1['high'].max() - w1['low'].min()) / w1['high'].max()) * 100
    d2 = ((w2['high'].max() - w2['low'].min()) / w2['high'].max()) * 100
    d3 = ((w3['high'].max() - w3['low'].min()) / w3['high'].max()) * 100

    # Điều kiện VCP: Biên độ thu hẹp dần và nhịp cuối cùng d3 phải cực kỳ chặt chẽ (<= 8%)
    is_vcp = (d1 > d2) and (d2 > d3) and (d3 <= 8.0) and (d1 <= 28.0)

    # 3. Kiểm tra Cạn Kiệt Thanh Khoản (Volume Dry-Up - VDU)
    vol_ma20 = float(df['VOL_MA20'].iloc[-1])
    last_3d_vol = float(df['volume'].iloc[-4:-1].mean())
    is_vdu = (last_3d_vol <= 0.65 * vol_ma20) if vol_ma20 > 0 else False

    # 4. Xác định Điểm Pivot & Quản trị rủi ro
    pivot_point = float(w3['high'].max())
    curr_vol = float(df['volume'].iloc[-1])
    is_breakout = (curr_price >= pivot_point) and (curr_vol >= 1.3 * vol_ma20)

    # Hard-stop: Tối đa 6.5% - 7.0%
    stop_loss = round(curr_price * 0.935, 2)
    risk_unit = curr_price - stop_loss
    target_price = round(curr_price + (risk_unit * 2.5), 2)  # R:R = 1:2.5

    result["is_uptrend"] = is_uptrend
    result["is_vcp"] = is_vcp
    result["is_breakout"] = is_breakout
    result["data"] = {
        "current_price": curr_price,
        "pivot_point": pivot_point,
        "stop_loss": stop_loss,
        "target_price": target_price,
        "contractions": f"-{d1:.1f}% | -{d2:.1f}% | -{d3:.1f}%",
        "vdu_ratio": round(last_3d_vol / vol_ma20, 2) if vol_ma20 > 0 else 0,
        "pct_from_52w_low": round(((curr_price - low_52w) / low_52w) * 100, 1),
        "pct_from_52w_high": round(((curr_price - high_52w) / high_52w) * 100, 1)
    }

    return is_uptrend, result

# ==============================================================================
# 4. MODULE CƠ SỞ DỮ LIỆU SQLITE (TRADING JOURNAL)
# ==============================================================================
DB_PATH = "trading_journal.db"

def init_sqlite_db():
    """Khởi tạo bảng quản lý nhật ký lệnh."""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS journal (
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
                pnl REAL DEFAULT 0,
                pnl_pct REAL DEFAULT 0
            )
        ''')
        conn.commit()
        conn.close()
    except Exception:
        pass

init_sqlite_db()

def log_trade(symbol: str, price: float, qty: int, sl: float, tp: float, reason: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    today = datetime.date.today().strftime("%Y-%m-%d")
    c.execute('''
        INSERT INTO journal (symbol, entry_date, entry_price, quantity, stop_loss, target_price, status, reason)
        VALUES (?, ?, ?, ?, ?, ?, 'OPEN', ?)
    ''', (symbol.upper(), today, price, qty, sl, tp, reason))
    conn.commit()
    conn.close()

def close_trade_record(trade_id: int, exit_price: float):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT entry_price, quantity FROM journal WHERE id = ?", (trade_id,))
    row = c.fetchone()
    if row:
        e_price, qty = row[0], row[1]
        pnl = (exit_price - e_price) * qty
        pnl_pct = ((exit_price - e_price) / e_price) * 100
        c.execute('''
            UPDATE journal
            SET exit_price = ?, status = 'CLOSED', pnl = ?, pnl_pct = ?
            WHERE id = ?
        ''', (exit_price, pnl, pnl_pct, trade_id))
        conn.commit()
    conn.close()

def fetch_journal() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM journal ORDER BY id DESC", conn)
    conn.close()
    return df

# ==============================================================================
# 5. GIAO DIỆN NGƯỜI DÙNG STREAMLIT
# ==============================================================================
st.sidebar.title("⚡ Minervini Hub")
st.sidebar.markdown("**Hệ thống Lọc VCP & Quản trị Rủi ro**")

default_list = "FPT, SSI, HPG, MWG, DGC, VHC, PVS, PVD, VCI, MBB, TCB, CTG, ACB, FRT"
watchlist_input = st.sidebar.text_area(
    "Danh sách mã quét:",
    value=default_list,
    height=120
)

# 3 Tab điều hướng chính
tab1, tab2, tab3 = st.tabs([
    "📈 Quét & Phân Tích Kỹ Thuật", 
    "⚖️ Quản Trị Vị Thế (Livermore)", 
    "📓 Sổ Lệnh Nhật Ký (SQLite)"
])

# ------------------------------------------------------------------------------
# TAB 1: BỘ LỌC VCP & ĐỒ THỊ STREAMLIT NATIVE
# ------------------------------------------------------------------------------
with tab1:
    st.subheader("Bộ Lọc Trend Template & Nhận Diện VCP Tự Động")

    col_btn1, col_btn2 = st.columns([2, 5])
    with col_btn1:
        start_scan = st.button("🚀 Quét Toàn Bộ Danh Sách", use_container_width=True)

    if start_scan:
        tickers = [t.strip().upper() for t in watchlist_input.split(",") if t.strip()]
        scan_results = []
        progress_bar = st.progress(0)

        for i, sym in enumerate(tickers):
            try:
                raw_df = get_tcbs_ohlcv(sym)
                if raw_df is not None and len(raw_df) >= 220:
                    df_calc = compute_technical_indicators(raw_df)
                    is_uptrend, details = evaluate_minervini_system(df_calc)
                    
                    if is_uptrend:
                        d = details["data"]
                        status = "⭐ PIVOT BREAKOUT" if details["is_breakout"] else ("🔥 VCP TÍCH LŨY" if details["is_vcp"] else "✅ UPTREND ĐẠT CHUẨN")
                        scan_results.append({
                            "Mã CP": sym,
                            "Trạng Thái": status,
                            "Giá HT": d["current_price"],
                            "Điểm Pivot": d["pivot_point"],
                            "Nhịp Siết VCP": d["contractions"],
                            "Tỷ Lệ Vol Cạn": f"{d['vdu_ratio']}x",
                            "Cắt Lỗ (-6.5%)": d["stop_loss"],
                            "Mục Tiêu (1:2.5)": d["target_price"],
                            "% Đáy 52W": f"+{d['pct_from_52w_low']}%"
                        })
            except Exception:
                pass
            progress_bar.progress((i + 1) / len(tickers))

        if scan_results:
            st.session_state["scan_data"] = scan_results
            st.success(f"Quét hoàn tất: Tìm thấy {len(scan_results)} mã đạt chuẩn Giai đoạn 2 (Uptrend)!")
        else:
            st.warning("Không có mã nào đạt tiêu chí Trend Template.")

    if "scan_data" in st.session_state and st.session_state["scan_data"]:
        st.dataframe(pd.DataFrame(st.session_state["scan_data"]), use_container_width=True)

    st.markdown("---")
    st.subheader("Biểu Đồ Kỹ Thuật Tinh Gọn (Native Line Chart)")
    
    chart_col1, chart_col2 = st.columns([2, 5])
    with chart_col1:
        inspect_ticker = st.text_input("Nhập mã xem biểu đồ:", value="FPT").upper()
        lookback = st.slider("Số phiên hiển thị:", min_value=60, max_value=250, value=120)

    if inspect_ticker:
        c_df = get_tcbs_ohlcv(inspect_ticker)
        if c_df is not None and len(c_df) >= 200:
            c_df = compute_technical_indicators(c_df)
            plot_df = c_df.iloc[-lookback:].copy()
            plot_df.set_index('date', inplace=True)

            st.write(f"**Cấu trúc Xu hướng Giá & MA50 / MA150 / MA200 - {inspect_ticker}**")
            # Sử dụng st.line_chart có sẵn của Streamlit
            st.line_chart(
                plot_df[['close', 'MA50', 'MA150', 'MA200']],
                height=350,
                use_container_width=True
            )

            st.write(f"**Khối Lượng & Đường Trung Bình Khối Lượng 20 Phiên**")
            # Sử dụng st.bar_chart có sẵn của Streamlit
            st.bar_chart(
                plot_df['volume'],
                height=150,
                use_container_width=True
            )
        else:
            st.info("Chưa có đủ dữ liệu lịch sử để hiển thị cho mã này.")

# ------------------------------------------------------------------------------
# TAB 2: QUẢN TRỊ RỦI RO & PHÂN BỔ VỊ THẾ (LIVERMORE 1-2% RULE)
# ------------------------------------------------------------------------------
with tab2:
    st.subheader("Máy Tính Định Lượng Vị Thế (Position Sizing Calculator)")
    st.markdown("""
    > **Nguyên tắc Quản trị Vốn của Mark Minervini:**
    > *Không bao giờ để một giao dịch đơn lẻ gây thua lỗ quá **1.0% - 2.0%** tổng NAV tài khoản. Mức dừng lỗ tối đa tuyệt đối là **7.0%**.*
    """)

    col_c1, col_c2, col_c3 = st.columns(3)
    with col_c1:
        nav = st.number_input("Tổng quy mô vốn (NAV) VNĐ:", min_value=10_000_000, value=500_000_000, step=50_000_000, format="%d")
    with col_c2:
        risk_pct = st.slider("Mức rủi ro chấp nhận (% NAV):", min_value=0.5, max_value=2.5, value=1.5, step=0.1)
    with col_c3:
        max_allowed_risk = nav * (risk_pct / 100)
        st.metric("Rủi ro tối đa cho phép / 1 lệnh", f"{max_allowed_risk:,.0f} VNĐ")

    st.markdown("---")
    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        entry_price = st.number_input("Giá mua kế hoạch (VNĐ):", value=45000, step=500)
    with col_p2:
        stop_price = st.number_input("Giá cắt lỗ (Stop-Loss VNĐ):", value=42000, step=500)
    with col_p3:
        target_price = st.number_input("Giá chốt lời dự kiến (VNĐ):", value=52500, step=500)

    if entry_price > stop_price:
        risk_per_share = entry_price - stop_price
        sl_pct = (risk_per_share / entry_price) * 100
        reward_per_share = target_price - entry_price
        rr_ratio = (reward_per_share / risk_per_share) if risk_per_share > 0 else 0

        # Số cổ phiếu tối đa = Rủi ro tối đa cho phép / Rủi ro trên 1 cổ phiếu
        max_shares = int(max_allowed_risk // risk_per_share)
        total_trade_value = max_shares * entry_price
        portfolio_exposure = (total_trade_value / nav) * 100

        st.markdown("### Kết Quả Tính Toán:")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Khối lượng mua tối đa", f"{max_shares:,.0f} CP")
        m2.metric("Tỷ trọng giải ngân / NAV", f"{portfolio_exposure:.1f}%")
        m3.metric("Tỷ lệ Cắt Lỗ", f"-{sl_pct:.2f}%", delta=f"-{sl_pct:.2f}%", delta_color="inverse")
        m4.metric("Tỷ lệ Risk : Reward", f"1 : {rr_ratio:.2f}")

        if sl_pct > 7.0:
            st.error(f"⚠️ Mức cắt lỗ (-{sl_pct:.2f}%) vượt trần an toàn 7% của Mark Minervini!")
        elif rr_ratio < 2.0:
            st.warning(f"⚠️ Tỷ lệ R:R (1:{rr_ratio:.2f}) nhỏ hơn 1:2.0 tiêu chuẩn.")
        else:
            st.success("✅ Vị thế đạt chuẩn quản trị rủi ro hoàn hảo!")

# ------------------------------------------------------------------------------
# TAB 3: NHẬT KÝ GIAO DỊCH SQLITE
# ------------------------------------------------------------------------------
with tab3:
    st.subheader("Nhật Ký Giao Dịch & Quản Lý Lệnh (SQLite)")

    with st.expander("➕ Ghi nhận vị thế vào lệnh mới", expanded=False):
        f1, f2, f3, f4 = st.columns(4)
        with f1:
            sym_in = st.text_input("Mã cổ phiếu:", "FPT").upper()
            price_in = st.number_input("Giá mua:", value=100000, step=500)
        with f2:
            qty_in = st.number_input("Khối lượng:", value=1000, step=100)
            sl_in = st.number_input("Giá cắt lỗ:", value=93500, step=500)
        with f3:
            tp_in = st.number_input("Giá chốt lời:", value=116000, step=500)
            reason_in = st.text_input("Lý do vào lệnh:", "Minervini VCP Pivot")
        with f4:
            st.write("")
            st.write("")
            if st.button("Lưu Vị Thế", use_container_width=True):
                if price_in > sl_in:
                    log_trade(sym_in, price_in, qty_in, sl_in, tp_in, reason_in)
                    st.success(f"Đã lưu lệnh {sym_in}!")
                    st.rerun()
                else:
                    st.error("Giá cắt lỗ phải nhỏ hơn giá vào lệnh!")

    # Truy vấn dữ liệu từ SQLite
    journal_df = fetch_journal()
    open_trades = journal_df[journal_df['status'] == 'OPEN'] if not journal_df.empty else pd.DataFrame()
    closed_trades = journal_df[journal_df['status'] == 'CLOSED'] if not journal_df.empty else pd.DataFrame()

    # Thống kê hiệu suất
    if not closed_trades.empty:
        total_closed = len(closed_trades)
        wins = closed_trades[closed_trades['pnl'] > 0]
        losses = closed_trades[closed_trades['pnl'] <= 0]
        win_rate = (len(wins) / total_closed) * 100
        net_pnl = closed_trades['pnl'].sum()

        s1, s2, s3 = st.columns(3)
        s1.metric("Tổng Số Lệnh Đã Đóng", f"{total_closed}")
        s2.metric("Tỷ Lệ Thắng (Win-Rate)", f"{win_rate:.1f}%")
        s3.metric("Tổng Lãi/Lỗ Thực Tế", f"{net_pnl:,.0f} VNĐ", delta=f"{net_pnl:,.0f}")

    st.markdown("---")
    st.markdown("#### Các Lệnh Đang Mở (Open Trades)")
    if not open_trades.empty:
        st.dataframe(open_trades[['id', 'symbol', 'entry_date', 'entry_price', 'quantity', 'stop_loss', 'target_price', 'reason']], use_container_width=True)

        col_c1, col_c2, col_c3 = st.columns([2, 2, 2])
        with col_c1:
            trade_select = st.selectbox("Chọn ID lệnh muốn đóng:", open_trades['id'].tolist())
        with col_c2:
            close_price_in = st.number_input("Giá bán đóng lệnh (VNĐ):", value=float(open_trades.loc[open_trades['id'] == trade_select, 'entry_price'].values[0]), step=500.0)
        with col_c3:
            st.write("")
            if st.button("🔒 Đóng Lệnh Ngay", use_container_width=True):
                close_trade_record(trade_select, close_price_in)
                st.success(f"Đã chốt lệnh #{trade_select}!")
                st.rerun()
    else:
        st.info("Hiện không có lệnh nào đang mở.")

    st.markdown("---")
    st.markdown("#### Lịch Sử Lệnh Đã Đóng (Closed Trades)")
    if not closed_trades.empty:
        st.dataframe(closed_trades[['id', 'symbol', 'entry_date', 'entry_price', 'exit_price', 'quantity', 'pnl', 'pnl_pct', 'reason']], use_container_width=True)
    else:
        st.caption("Chưa có lịch sử lệnh đã đóng.")