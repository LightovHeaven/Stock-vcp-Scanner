Mã nguồn hoàn chỉnh của ứng dụng:

```python
import streamlit as st
import pandas as pd
import numpy as np
import requests
import sqlite3
import datetime
import time
from typing import Optional, Dict, Any

# ==============================================================================
# 1. CẤU HÌNH GIAO DIỆN STREAMLIT
# ==============================================================================
st.set_page_config(
    page_title="Minervini & Livermore Quant Terminal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# 2. MODULE CƠ SỞ DỮ LIỆU: NHẬT KÝ GIAO DỊCH (SQLITE)
# ==============================================================================
DB_NAME = "trading_journal.db"

def init_db():
    """Khởi tạo cấu trúc bảng SQLite nếu chưa tồn tại."""
    conn = sqlite3.connect(DB_NAME)
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
            status TEXT NOT NULL, -- 'OPEN' hoặc 'CLOSED'
            reason TEXT,
            pnl REAL,
            pnl_pct REAL
        )
    ''')
    conn.commit()
    conn.close()

init_db()

def add_trade_to_db(symbol: str, entry_price: float, qty: int, stop_loss: float, target_price: float, reason: str):
    """Lưu vị thế mua mới vào nhật ký."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    today_str = datetime.date.today().strftime("%Y-%m-%d")
    c.execute('''
        INSERT INTO trades (symbol, entry_date, entry_price, quantity, stop_loss, target_price, status, reason, pnl, pnl_pct)
        VALUES (?, ?, ?, ?, ?, ?, 'OPEN', ?, 0, 0)
    ''', (symbol.upper(), today_str, entry_price, qty, stop_loss, target_price, reason))
    conn.commit()
    conn.close()

def close_trade_in_db(trade_id: int, exit_price: float):
    """Đóng lệnh giao dịch và tự động hạch toán PnL."""
    conn = sqlite3.connect(DB_NAME)
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

def get_all_trades() -> pd.DataFrame:
    """Truy vấn toàn bộ dữ liệu lịch sử lệnh."""
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query("SELECT * FROM trades ORDER BY id DESC", conn)
    conn.close()
    return df

# ==============================================================================
# 3. MODULE CẢNH BÁO TELEGRAM (BOT NOTIFIER)
# ==============================================================================
def send_telegram_alert(bot_token: str, chat_id: str, message: str) -> bool:
    """Gửi thông báo Markdown về Telegram Bot."""
    if not bot_token or not chat_id:
        return False
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "Markdown"
    }
    try:
        response = requests.post(url, json=payload, timeout=8)
        return response.status_code == 200
    except Exception as e:
        st.error(f"Lỗi khi gửi cảnh báo Telegram: {e}")
        return False

# ==============================================================================
# 4. MODULE DỮ LIỆU THỊ TRƯỜNG (VN-STOCK DATA FETCHING)
# ==============================================================================
@st.cache_data(ttl=3600)
def fetch_stock_data(symbol: str, days: int = 400) -> Optional[pd.DataFrame]:
    """Lấy dữ liệu giá lịch sử OHLCV qua API công khai TCBS."""
    end_time = int(time.time())
    start_time = int(end_time - (days * 86400))
    url = f"https://apipubaws.tcbs.com.vn/stock-insight/v1/stock/bars-long-term?ticker={symbol.upper()}&type=stock&resolution=D&from={start_time}&to={end_time}"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return None
        data = response.json()
        if "data" not in data or not data["data"]:
            return None

        df = pd.DataFrame(data["data"])
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
        return df
    except Exception:
        return None

# ==============================================================================
# 5. THUẬT TOÁN ĐỊNH LƯỢNG: TREND TEMPLATE & PIVOT BREAKOUT
# ==============================================================================
def analyze_minervini_trend(df: pd.DataFrame) -> Dict[str, Any]:
    """Đánh giá điều kiện Trend Template và Điểm mua bứt phá nền tích lũy."""
    res = {"pass_trend": False, "pass_pivot": False, "metrics": {}}

    if len(df) < 220:
        return res

    # 1. Đường trung bình động
    df['MA50'] = df['close'].rolling(window=50).mean()
    df['MA150'] = df['close'].rolling(window=150).mean()
    df['MA200'] = df['close'].rolling(window=200).mean()
    df['VOL_MA20'] = df['volume'].rolling(window=20).mean()

    current_price = df['close'].iloc[-1]
    ma50 = df['MA50'].iloc[-1]
    ma150 = df['MA150'].iloc[-1]
    ma200 = df['MA200'].iloc[-1]
    ma200_22d_ago = df['MA200'].iloc[-22]

    # 2. Đỉnh/Đáy 52 tuần
    period_52w = min(252, len(df))
    high_52w = df['high'].iloc[-period_52w:].max()
    low_52w = df['low'].iloc[-period_52w:].min()

    # 3. Tiêu chuẩn Trend Template (Minervini)
    c1 = (current_price > ma50) and (current_price > ma150) and (current_price > ma200)
    c2 = (ma50 > ma150) and (ma150 > ma200)
    c3 = ma200 > ma200_22d_ago
    c4 = current_price >= (low_52w * 1.25)
    c5 = current_price >= (high_52w * 0.80)

    trend_passed = all([c1, c2, c3, c4, c5])

    # 4. Xác định Pivot Breakout (Jesse Livermore)
    base_window = df.iloc[-16:-1]
    pivot_resistance = base_window['high'].max()
    current_vol = df['volume'].iloc[-1]
    vol_ma20 = df['VOL_MA20'].iloc[-1]

    is_breakout = (current_price >= pivot_resistance) and (current_vol >= 1.4 * vol_ma20)

    # 5. Quản trị rủi ro cố định
    stop_loss = round(current_price * 0.935, 2)
    risk_per_share = current_price - stop_loss
    target_price = round(current_price + (risk_per_share * 2.5), 2)

    res["pass_trend"] = trend_passed
    res["pass_pivot"] = is_breakout
    res["metrics"] = {
        "current_price": current_price,
        "ma50": ma50,
        "ma150": ma150,
        "ma200": ma200,
        "pivot_resistance": pivot_resistance,
        "stop_loss": stop_loss,
        "target_price": target_price,
        "risk_reward": 2.5,
        "vol_ratio": round(current_vol / vol_ma20, 2) if vol_ma20 > 0 else 0,
        "pct_from_52w_low": round(((current_price - low_52w) / low_52w) * 100, 2),
        "pct_from_52w_high": round(((current_price - high_52w) / high_52w) * 100, 2)
    }
    return res

# ==============================================================================
# 6. GIAO DIỆN STREAMLIT TERMINAL
# ==============================================================================
st.sidebar.title("🛡️ Cấu Hình Hệ Thống")

with st.sidebar.expander("📲 Cấu hình Telegram Bot", expanded=False):
    tg_token = st.text_input("Bot Token", type="password", help="Nhận từ @BotFather")
    tg_chat_id = st.text_input("Chat ID", help="Nhận từ @userinfobot")
    if st.button("Kiểm tra kết nối Bot"):
        if send_telegram_alert(tg_token, tg_chat_id, "🔔 *Kết nối Hệ thống Quant Trading thành công!*"):
            st.success("Đã gửi thông báo thành công!")
        else:
            st.error("Gửi thất bại! Hãy kiểm tra Token và Chat ID.")

default_tickers = "SSI, VND, FPT, HPG, MWG, DGC, VHC, PVS, PVD, VCI, MBB, TCB, CTG, ACB, FRT"
selected_tickers = st.sidebar.text_area(
    "Danh sách mã quét:",
    value=default_tickers,
    height=120
)

tab1, tab2, tab3 = st.tabs([
    "🔍 Bộ Lọc Minervini & Livermore", 
    "📓 Quản Trị Vốn & Nhật Ký Giao Dịch", 
    "📊 Hiệu Suất Danh Mục"
])

# TAB 1: BỘ LỌC CHIẾN LƯỢC
with tab1:
    st.subheader("Bộ Lọc Trend Template & Tín Hiệu Điểm Mua Pivot")
    if st.button("🚀 Bắt đầu Quét Toàn Bộ Danh Sách", use_container_width=True):
        tickers = [t.strip().upper() for t in selected_tickers.split(",") if t.strip()]
        progress_bar = st.progress(0)
        results = []

        for idx, ticker in enumerate(tickers):
            df = fetch_stock_data(ticker)
            if df is not None:
                analysis = analyze_minervini_trend(df)
                if analysis["pass_trend"]:
                    m = analysis["metrics"]
                    status_text = "🔥 PIVOT BREAKOUT" if analysis["pass_pivot"] else "✅ UPTREND ĐẠT CHUẨN"
                    
                    results.append({
                        "Mã CP": ticker,
                        "Trạng Thái": status_text,
                        "Giá Hiện Tại": m["current_price"],
                        "Pivot Kháng Cự": m["pivot_resistance"],
                        "Cắt Lỗ (-6.5%)": m["stop_loss"],
                        "Mục Tiêu R:R 1:2.5": m["target_price"],
                        "Khối Lượng / MA20": f"{m['vol_ratio']}x",
                        "% Từ Đáy 52W": f"+{m['pct_from_52w_low']}%",
                        "% Từ Đỉnh 52W": f"{m['pct_from_52w_high']}%"
                    })
                    
                    if analysis["pass_pivot"] and tg_token and tg_chat_id:
                        msg = (
                            f"🚨 *TÍN HIỆU BREAKOUT ĐẠT CHUẨN MINERVINI*\n\n"
                            f"📌 *Cổ phiếu:* `{ticker}`\n"
                            f"💵 *Giá:* `{m['current_price']:,} VNĐ`\n"
                            f"⚡ *Vượt Pivot:* `{m['pivot_resistance']:,} VNĐ`\n"
                            f"📊 *Vol nổ:* `{m['vol_ratio']}x* MA20\n"
                            f"🛑 *Cắt lỗ:* `{m['stop_loss']:,} VNĐ`\n"
                            f"🎯 *Mục tiêu:* `{m['target_price']:,} VNĐ`\n"
                            f"⏱ *Thời gian:* {datetime.datetime.now().strftime('%H:%M:%S %d/%m/%Y')}"
                        )
                        send_telegram_alert(tg_token, tg_chat_id, msg)
            
            progress_bar.progress((idx + 1) / len(tickers))

        if results:
            st.success(f"Tìm thấy {len(results)} mã đạt chuẩn!")
            st.dataframe(pd.DataFrame(results), use_container_width=True)
        else:
            st.warning("Không có mã nào đạt tiêu chí Trend Template.")

# TAB 2: QUẢN TRỊ DANH MỤC & VỊ THẾ
with tab2:
    st.subheader("Tính Toán Khối Lượng Giao Dịch (Position Sizing)")
    col_v1, col_v2, col_v3 = st.columns(3)
    with col_v1:
        total_nav = st.number_input("Tổng quy mô vốn (NAV) VNĐ:", value=500000000, step=50000000)
    with col_v2:
        risk_pct = st.slider("Rủi ro tối đa mỗi lệnh (% NAV):", min_value=0.5, max_value=2.5, value=1.5, step=0.1)
    with col_v3:
        max_risk_amount = total_nav * (risk_pct / 100)
        st.metric("Rủi ro cho phép/lệnh", f"{max_risk_amount:,.0f} VNĐ")

    st.markdown("---")
    st.subheader("Ghi Nhận Lệnh Mua")
    col_t1, col_t2, col_t3, col_t4 = st.columns(4)
    with col_t1:
        trade_ticker = st.text_input("Mã Cổ Phiếu:", "SSI").upper()
        trade_entry = st.number_input("Giá Mua Kế Hoạch (VNĐ):", value=35000, step=500)
    with col_t2:
        trade_sl = st.number_input("Giá Cắt Lỗ (VNĐ):", value=32700, step=500)
        risk_per_unit = max(1.0, trade_entry - trade_sl)
        recommended_qty = int(max_risk_amount // risk_per_unit)
    with col_t3:
        trade_tp = st.number_input("Mục Tiêu Chốt Lời (VNĐ):", value=40750, step=500)
        trade_qty = st.number_input("Khối lượng dự kiến mua:", value=recommended_qty, step=100)
    with col_t4:
        trade_reason = st.text_input("Lý do vào lệnh:", "Minervini VCP Pivot Breakout")
        st.write("")
        if st.button("💾 Lưu Nhật Ký", use_container_width=True):
            if trade_entry > trade_sl:
                add_trade_to_db(trade_ticker, trade_entry, trade_qty, trade_sl, trade_tp, trade_reason)
                st.success(f"Đã lưu lệnh {trade_ticker}!")
            else:
                st.error("Giá cắt lỗ phải nhỏ hơn giá mua!")

    st.markdown("---")
    st.subheader("Các Vị Thế Đang Mở (Open Positions)")
    trades_df = get_all_trades()
    open_trades = trades_df[trades_df['status'] == 'OPEN'] if not trades_df.empty else pd.DataFrame()

    if not open_trades.empty:
        st.dataframe(open_trades[['id', 'symbol', 'entry_date', 'entry_price', 'quantity', 'stop_loss', 'target_price', 'reason']], use_container_width=True)
        col_c1, col_c2, col_c3 = st.columns([2, 2, 2])
        with col_c1:
            close_id = st.selectbox("Chọn ID lệnh cần đóng:", open_trades['id'].tolist())
        with col_c2:
            exit_price = st.number_input("Giá bán thực tế (VNĐ):", value=38000, step=500)
        with col_c3:
            st.write("")
            if st.button("🔒 Đóng Vị Thế", use_container_width=True):
                close_trade_in_db(close_id, exit_price)
                st.success(f"Đã đóng lệnh #{close_id}!")
                st.rerun()
    else:
        st.info("Hiện không có vị thế nào đang mở.")

# TAB 3: HIỆU SUẤT ĐỊNH LƯỢNG
with tab3:
    st.subheader("Thống Kê Hiệu Suất Hệ Thống")
    closed_trades = trades_df[trades_df['status'] == 'CLOSED'] if not trades_df.empty else pd.DataFrame()

    if not closed_trades.empty:
        total_closed = len(closed_trades)
        wins = closed_trades[closed_trades['pnl'] > 0]
        losses = closed_trades[closed_trades['pnl'] <= 0]
        
        win_rate = (len(wins) / total_closed) * 100 if total_closed > 0 else 0
        total_pnl = closed_trades['pnl'].sum()
        avg_win = wins['pnl'].mean() if not wins.empty else 0
        avg_loss = abs(losses['pnl'].mean()) if not losses.empty else 1
        rr_ratio = (avg_win / avg_loss) if avg_loss != 0 else 0

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Lệnh Đã Đóng", f"{total_closed}")
        col_m2.metric("Tỷ Lệ Thắng (Win-Rate)", f"{win_rate:.1f}%")
        col_m3.metric("R:R Thực Tế", f"1:{rr_ratio:.2f}")
        col_m4.metric("Lợi Nhuận Ròng", f"{total_pnl:,.0f} VNĐ", delta=f"{total_pnl:,.0f}")

        st.markdown("---")
        st.dataframe(closed_trades[['id', 'symbol', 'entry_date', 'entry_price', 'exit_price', 'quantity', 'pnl', 'pnl_pct', 'reason']], use_container_width=True)
    else:
        st.info("Chưa có lệnh nào được đóng trong hệ thống.")
