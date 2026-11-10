"""
Ứng dụng Web: Realtime Vietnam Stock Screener (Minervini SEPA + Livermore Pivot)
Tác giả: Senior Software Engineer & Pro Trader
Công nghệ: Python 3.10+, Streamlit, Plotly, Pandas, TCBS Public API
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import io

# ==============================================================================
# CẤU HÌNH TRANG STREAMLIT
# ==============================================================================
st.set_page_config(
    page_title="VN-Stock Screener (SEPA & Pivot)",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Header và phong cách giao diện
st.markdown("""
    <style>
    .main-title {
        font-size: 28px;
        font-weight: 800;
        color: #1E88E5;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 15px;
        color: #616161;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 12px;
        border: 1px solid #e0e0e0;
    }
    </style>
""", unsafe_allow_html=True)


# ==============================================================================
# DANH SÁCH MÃ CHỨNG KHOÁN MẪU (TOP THANH KHOẢN & DẪN DẮT THỊ TRƯỜNG)
# ==============================================================================
VN30_LIST = [
    "ACB", "BCM", "BID", "BVH", "CTG", "FPT", "GAS", "GVR", "HDB", "HPG", 
    "MBB", "MSN", "MWG", "PLX", "POW", "SAB", "SHB", "SSB", "SSI", "STB", 
    "TCB", "TPB", "VCB", "VHM", "VIB", "VIC", "VJC", "VNM", "VPB", "VRE"
]

GROWTH_WATCHLIST = VN30_LIST + [
    "DGC", "FRT", "PVD", "PVS", "KDH", "NLG", "KBC", "DIG", "DXG", 
    "VIX", "VND", "HCM", "DCM", "DPM", "HSG", "NKG", "GEX", "PC1", "ANV"
]

# Loại bỏ trùng lặp và sắp xếp
GROWTH_WATCHLIST = sorted(list(set(GROWTH_WATCHLIST)))


# ==============================================================================
# MODULE TRUY XUẤT DỮ LIỆU CHỨNG KHOÁN (TCBS API)
# ==============================================================================
class TCBSDataProvider:
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json"
    }

    @staticmethod
    def get_price_history(symbol: str, days_back: int = 400) -> pd.DataFrame:
        """Lấy lịch sử nến ngày (OHLCV) từ TCBS."""
        try:
            end_ts = int(time.time())
            start_ts = end_ts - (days_back * 86400)
            url = f"https://apipubaws.tcbs.com.vn/stock-insight/v1/stock/bars-long-term?ticker={symbol}&type=stock&resolution=D&from={start_ts}&to={end_ts}"
            
            response = requests.get(url, headers=TCBSDataProvider.HEADERS, timeout=8)
            if response.status_code != 200:
                return pd.DataFrame()
            
            data = response.json().get("data", [])
            if not data:
                return pd.DataFrame()
            
            df = pd.DataFrame(data)
            df['tradingDate'] = pd.to_datetime(df['tradingDate'])
            df = df.sort_values('tradingDate').reset_index(drop=True)
            
            # Chuẩn hóa cột
            df = df[['tradingDate', 'open', 'high', 'low', 'close', 'volume']]
            df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            
            # Chia giá về đúng đơn vị nghìn đồng nếu dữ liệu thô nhân 1000
            if df['Close'].iloc[-1] > 1000:
                df[['Open', 'High', 'Low', 'Close']] = df[['Open', 'High', 'Low', 'Close']] / 1000.0
                
            return df
        except Exception:
            return pd.DataFrame()

    @staticmethod
    def get_financial_ratios(symbol: str) -> dict:
        """Lấy chỉ số tài chính (EPS, Doanh thu, ROE) theo quý từ TCBS."""
        default_res = {
            "roe": 0.0,
            "eps_growth_yoy": 0.0,
            "rev_growth_yoy": 0.0,
            "quarter": "N/A"
        }
        try:
            url = f"https://apipubaws.tcbs.com.vn/tcanalysis/v1/finance/{symbol}/financialratio?yearly=0&isAll=false"
            response = requests.get(url, headers=TCBSDataProvider.HEADERS, timeout=6)
            if response.status_code != 200:
                return default_res

            items = response.json()
            if not isinstance(items, list) or len(items) == 0:
                return default_res

            # Sắp xếp quý mới nhất lên đầu
            items = sorted(items, key=lambda x: (x.get('year', 0), x.get('quarter', 0)), reverse=True)
            latest = items[0]
            
            # ROE (TCBS trả về theo % vd 18.5)
            roe = float(latest.get('roe', 0) or 0)
            if roe <= 1.0 and roe > 0:
                roe = roe * 100  # Quy đổi nếu tỷ lệ thập phân

            # Tăng trưởng EPS và Doanh thu so với cùng kỳ năm trước (4 quý trước)
            eps_growth = 0.0
            rev_growth = 0.0
            
            if len(items) >= 5:
                prev_year_quarter = items[4] # Quý cùng kỳ năm ngoái
                
                # EPS growth
                curr_eps = float(latest.get('earningPerShare', 0) or 0)
                prev_eps = float(prev_year_quarter.get('earningPerShare', 0) or 0)
                if prev_eps > 0:
                    eps_growth = ((curr_eps - prev_eps) / abs(prev_eps)) * 100
                
                # Doanh thu / Lợi nhuận gộp growth
                curr_rev = float(latest.get('netRevenue', 0) or latest.get('revenue', 0) or 0)
                prev_rev = float(prev_year_quarter.get('netRevenue', 0) or prev_year_quarter.get('revenue', 0) or 0)
                if prev_rev > 0:
                    rev_growth = ((curr_rev - prev_rev) / abs(prev_rev)) * 100

            q_label = f"Q{latest.get('quarter', '?')}/{latest.get('year', '?')}"
            
            return {
                "roe": round(roe, 2),
                "eps_growth_yoy": round(eps_growth, 2),
                "rev_growth_yoy": round(rev_growth, 2),
                "quarter": q_label
            }
        except Exception:
            return default_res


# ==============================================================================
# BỘ TÍNH TOÁN BỘ LỌC KỸ THUẬT & CƠ BẢN
# ==============================================================================
def analyze_stock(symbol: str, params: dict) -> dict:
    """Phân tích kỹ thuật SEPA Minervini + Pivot Livermore kết hợp chỉ số FA."""
    df = TCBSDataProvider.get_price_history(symbol, days_back=380)
    
    # Yêu cầu tối thiểu 210 phiên giao dịch để tính MA200 và độ dốc 20 phiên
    if df.empty or len(df) < 210:
        return None

    # Tính toán các chỉ báo Kỹ thuật cơ bản
    df['MA50'] = df['Close'].rolling(window=params['ma_fast']).mean()
    df['MA150'] = df['Close'].rolling(window=params['ma_mid']).mean()
    df['MA200'] = df['Close'].rolling(window=params['ma_slow']).mean()
    df['MA20_Vol'] = df['Volume'].rolling(window=20).mean()

    # Tính True Range và ATR để đánh giá biên độ nén giá VCP
    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
    df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df['ATR5'] = df['TR'].rolling(window=5).mean()
    df['ATR20'] = df['TR'].rolling(window=20).mean()

    # Dữ liệu phiên hiện tại và quá khứ
    curr = df.iloc[-1]
    prev = df.iloc[-2]
    ma200_20d_ago = df['MA200'].iloc[-21] if len(df) >= 221 else df['MA200'].iloc[0]

    close_p = curr['Close']
    change_pct = ((close_p - prev['Close']) / prev['Close']) * 100

    # Đỉnh/Đáy 52 tuần (~250 phiên gần nhất)
    lookback_52w = min(len(df), 250)
    high_52w = df['High'].iloc[-lookback_52w:].max()
    low_52w = df['Low'].iloc[-lookback_52w:].min()

    pct_above_low = ((close_p - low_52w) / low_52w) * 100
    pct_below_high = ((high_52w - close_p) / high_52w) * 100

    # 1. TIÊU CHÍ TREND TEMPLATE (MINERVINI SEPA)
    cond_ma_stack = (close_p > curr['MA50']) and (curr['MA50'] > curr['MA150']) and (curr['MA150'] > curr['MA200'])
    cond_ma200_up = curr['MA200'] > ma200_20d_ago
    cond_52w_low = pct_above_low >= params['min_from_low']
    cond_52w_high = pct_below_high <= params['max_from_high']
    
    pass_trend_template = cond_ma_stack and cond_ma200_up and cond_52w_low and cond_52w_high

    # 2. TIÊU CHÍ VCP & PIVOT BREAKOUT (LIVERMORE & MINERVINI)
    # VCP: Biên độ 5 phiên gần nhất thu hẹp đáng kể so với bình quân 20 phiên
    is_vcp = curr['ATR5'] < (curr['ATR20'] * 0.75)

    # Volume Dry-up: Trung bình 3 phiên gần nhất cạn kiệt (< 80% MA20 Volume)
    avg_vol_3d = df['Volume'].iloc[-3:].mean()
    is_vol_dryup = avg_vol_3d < (0.8 * curr['MA20_Vol'])

    # Pivot / Pocket Breakout: Phiên nay tăng mạnh kèm vol vượt trội
    is_breakout = (change_pct >= params['breakout_price_pct']) and (curr['Volume'] >= (params['breakout_vol_mult'] * curr['MA20_Vol']))

    # 3. CHỈ SỐ CƠ BẢN (FA)
    fa_data = {"roe": 0.0, "eps_growth_yoy": 0.0, "rev_growth_yoy": 0.0, "quarter": "N/A"}
    if params['use_fa_filter']:
        fa_data = TCBSDataProvider.get_financial_ratios(symbol)

    pass_fa = True
    if params['use_fa_filter']:
        pass_fa = (
            fa_data['eps_growth_yoy'] >= params['min_eps_growth'] and
            fa_data['rev_growth_yoy'] >= params['min_rev_growth'] and
            fa_data['roe'] >= params['min_roe']
        )

    # ĐIỂM KỸ THUẬT TỔNG HỢP (TECHNICAL SCORE - Thang 100)
    score = 0
    if cond_ma_stack: score += 30
    if cond_ma200_up: score += 15
    if cond_52w_low and cond_52w_high: score += 15
    if is_vcp or is_vol_dryup: score += 15
    if is_breakout: score += 25

    return {
        "Symbol": symbol,
        "Close": round(close_p, 2),
        "Change_%": round(change_pct, 2),
        "Volume": int(curr['Volume']),
        "MA50": round(curr['MA50'], 2),
        "MA150": round(curr['MA150'], 2),
        "MA200": round(curr['MA200'], 2),
        "High_52W": round(high_52w, 2),
        "Low_52W": round(low_52w, 2),
        "Cách_Đỉnh_%": round(pct_below_high, 1),
        "Trên_Đáy_%": round(pct_above_low, 1),
        "Trend_Template": pass_trend_template,
        "VCP_VolDryup": (is_vcp or is_vol_dryup),
        "Pivot_Breakout": is_breakout,
        "Pass_FA": pass_fa,
        "Score": score,
        "ROE_%": fa_data['roe'],
        "EPS_Growth_%": fa_data['eps_growth_yoy'],
        "Rev_Growth_%": fa_data['rev_growth_yoy'],
        "FA_Quarter": fa_data['quarter']
    }


# ==============================================================================
# HÀM VẼ BIỂU ĐỒ NẾN KỸ THUẬT CHUYÊN NGHIỆP (PLOTLY)
# ==============================================================================
def plot_stock_chart(symbol: str):
    """Vẽ biểu đồ tương tác Nến Nhật, Khối lượng và các đường MA."""
    df = TCBSDataProvider.get_price_history(symbol, days_back=260)
    if df.empty:
        st.warning(f"Không thể tải đồ thị kỹ thuật cho mã {symbol}.")
        return

    df['MA50'] = df['Close'].rolling(50).mean()
    df['MA150'] = df['Close'].rolling(150).mean()
    df['MA200'] = df['Close'].rolling(200).mean()
    df['MA20_Vol'] = df['Volume'].rolling(20).mean()

    # Màu nến theo tiêu chuẩn Việt Nam (Xanh lá / Đỏ)
    colors = ['#00C076' if row['Close'] >= row['Open'] else '#FF3B30' for _, row in df.iterrows()]

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.7, 0.3],
        specs=[[{"type": "candlestick"}], [{"type": "bar"}]]
    )

    # Nến Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df['Date'],
            open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
            name="Giá",
            increasing_line_color='#00C076', decreasing_line_color='#FF3B30'
        ),
        row=1, col=1
    )

    # Các đường MA
    fig.add_trace(go.Scatter(x=df['Date'], y=df['MA50'], line=dict(color='#2962FF', width=1.5), name="MA50 (Ngắn hạn)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['MA150'], line=dict(color='#FF6D00', width=1.5), name="MA150 (Trung hạn)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['MA200'], line=dict(color='#D500F9', width=2), name="MA200 (Dài hạn)"), row=1, col=1)

    # Cột Volume và MA20 Volume
    fig.add_trace(go.Bar(x=df['Date'], y=df['Volume'], marker_color=colors, name="Khối lượng"), row=2, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['MA20_Vol'], line=dict(color='#FFA000', width=1.2), name="MA20 Khối lượng"), row=2, col=1)

    fig.update_layout(
        title=f"ĐỒ THỊ KỸ THUẬT CỔ PHIẾU {symbol} - MINERVINI SETUP",
        yaxis_title="Giá (1,000 VND)",
        yaxis2_title="Khối lượng",
        xaxis_rangeslider_visible=False,
        height=580,
        margin=dict(l=40, r=40, t=50, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, use_container_width=True)


# ==============================================================================
# GIAO DIỆN CHÍNH (STREAMLIT UI)
# ==============================================================================
st.markdown('<div class="main-title">⚡ REALTIME VIETNAM STOCK SCREENER</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Hệ thống lọc cổ phiếu tăng trưởng theo phương pháp Minervini SEPA & Điểm nổ Livermore Pivot</div>', unsafe_allow_html=True)

# --- SIDEBAR: CẤU HÌNH BỘ LỌC ---
with st.sidebar:
    st.header("⚙️ THIẾT LẬP BỘ LỌC")
    
    universe_choice = st.selectbox(
        "🎯 Danh sách theo dõi:",
        ["Top Cổ Phiếu Thanh Khoản (45 mã)", "VN30 Index (30 mã lớn)", "Tùy nhập danh sách"]
    )
    
    if universe_choice == "Top Cổ Phiếu Thanh Khoản (45 mã)":
        target_tickers = GROWTH_WATCHLIST
    elif universe_choice == "VN30 Index (30 mã lớn)":
        target_tickers = VN30_LIST
    else:
        custom_input = st.text_area("Nhập các mã (ngăn cách bằng dấu phẩy):", "FPT, HPG, SSI, DGC, MWG, TCB, VHM")
        target_tickers = [x.strip().upper() for x in custom_input.split(",") if x.strip()]

    st.markdown("---")
    st.subheader("1. Bộ Tiêu Chí Kỹ Thuật (SEPA)")
    require_trend = st.checkbox("Bắt buộc đạt Trend Template (MA50>150>200)", value=True)
    require_pivot = st.checkbox("Ưu tiên điểm bùng nổ (Pocket/Pivot Breakout)", value=False)
    
    col_sb1, col_sb2 = st.columns(2)
    with col_sb1:
        min_from_low = st.number_input("% Trên Đáy 52W:", min_value=0, max_value=200, value=25)
        breakout_price_pct = st.number_input("% Tăng giá Breakout:", min_value=1.0, max_value=10.0, value=2.5, step=0.5)
    with col_sb2:
        max_from_high = st.number_input("% Cách Đỉnh 52W:", min_value=5, max_value=100, value=25)
        breakout_vol_mult = st.number_input("Tỷ lệ Vol bùng nổ:", min_value=1.0, max_value=5.0, value=1.4, step=0.2)

    st.markdown("---")
    st.subheader("2. Bộ Tiêu Chí Cơ Bản (FA)")
    use_fa_filter = st.checkbox("Kích hoạt lọc cơ bản (FA)", value=False)
    
    min_eps_growth = 20.0
    min_rev_growth = 15.0
    min_roe = 15.0
    if use_fa_filter:
        min_eps_growth = st.slider("Tăng trưởng EPS YoY tối thiểu (%):", 0, 100, 20)
        min_rev_growth = st.slider("Tăng trưởng Doanh thu YoY (%):", 0, 100, 15)
        min_roe = st.slider("ROE tối thiểu (%):", 0, 50, 15)

    params = {
        "ma_fast": 50,
        "ma_mid": 150,
        "ma_slow": 200,
        "min_from_low": min_from_low,
        "max_from_high": max_from_high,
        "breakout_price_pct": breakout_price_pct,
        "breakout_vol_mult": breakout_vol_mult,
        "use_fa_filter": use_fa_filter,
        "min_eps_growth": min_eps_growth,
        "min_rev_growth": min_rev_growth,
        "min_roe": min_roe
    }

    start_scan_btn = st.button("🚀 BẮT ĐẦU LỌC CỔ PHIẾU", type="primary", use_container_width=True)

# Khởi tạo state kết quả nếu chưa có
if "scan_results" not in st.session_state:
    st.session_state.scan_results = None

# --- XỬ LÝ LỌC CỔ PHIẾU ---
if start_scan_btn:
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text(f"Đang phân tích {len(target_tickers)} mã cổ phiếu...")

    results = []
    completed = 0
    total = len(target_tickers)

    # Dùng đa luồng (10 worker) để quét song song với tốc độ cao
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_stock = {executor.submit(analyze_stock, ticker, params): ticker for ticker in target_tickers}
        for future in as_completed(future_to_stock):
            res = future.result()
            if res:
                results.append(res)
            completed += 1
            progress_bar.progress(completed / total)

    status_text.text("Phân tích hoàn tất!")
    progress_bar.empty()
    status_text.empty()
    st.session_state.scan_results = results

# --- HIỂN THỊ KẾT QUẢ ---
if st.session_state.scan_results:
    raw_df = pd.DataFrame(st.session_state.scan_results)
    
    # Áp dụng các điều kiện lọc theo cấu hình
    filtered_df = raw_df.copy()
    if require_trend:
        filtered_df = filtered_df[filtered_df['Trend_Template'] == True]
    if require_pivot:
        filtered_df = filtered_df[filtered_df['Pivot_Breakout'] == True]
    if use_fa_filter:
        filtered_df = filtered_df[filtered_df['Pass_FA'] == True]

    # Sắp xếp theo điểm kỹ thuật giảm dần
    filtered_df = filtered_df.sort_values(by=["Score", "Change_%"], ascending=[False, False]).reset_index(drop=True)

    st.subheader(f"📊 KẾT QUẢ QUÉT: Tìm thấy {len(filtered_df)} / {len(raw_df)} cổ phiếu đạt điều kiện")

    if not filtered_df.empty:
        # Bảng hiển thị giao diện đẹp mắt
        display_cols = [
            "Symbol", "Close", "Change_%", "Volume", "Cách_Đỉnh_%", "Trên_Đáy_%",
            "Trend_Template", "VCP_VolDryup", "Pivot_Breakout", "Score"
        ]
        if use_fa_filter:
            display_cols += ["ROE_%", "EPS_Growth_%", "Rev_Growth_%", "FA_Quarter"]

        styled_df = filtered_df[display_cols].copy()
        
        # Định dạng icon cho các cột tín hiệu
        styled_df['Trend_Template'] = styled_df['Trend_Template'].apply(lambda x: "🟢 Đạt" if x else "⚪ Không")
        styled_df['VCP_VolDryup'] = styled_df['VCP_VolDryup'].apply(lambda x: "🔥 Cạn/VCP" if x else "-")
        styled_df['Pivot_Breakout'] = styled_df['Pivot_Breakout'].apply(lambda x: "🚀 Pivot Nổ" if x else "-")

        st.dataframe(
            styled_df.style.format({
                "Close": "{:.2f}",
                "Change_%": "{:+.2f}%",
                "Volume": "{:,.0f}",
                "Cách_Đỉnh_%": "-{:.1f}%",
                "Trên_Đáy_%": "+{:.1f}%",
                "Score": "{:.0f}/100"
            }),
            use_container_width=True,
            height=300
        )

        # Xuất file Excel / CSV
        col_dl1, col_dl2 = st.columns([1, 5])
        with col_dl1:
            csv_buffer = io.StringIO()
            styled_df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
            st.download_button(
                label="📥 Tải kết quả CSV",
                data=csv_buffer.getvalue(),
                file_name=f"stock_screener_{time.strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )

        st.markdown("---")
        # --- PHẦN CHI TIẾT & BIỂU ĐỒ NẾN CHO TỪNG CỔ PHIẾU ---
        st.subheader("🔍 PHÂN TÍCH KỸ THUẬT CHI TIẾT THEO MÃ")
        
        col_sel1, col_sel2 = st.columns([2, 5])
        with col_sel1:
            selected_symbol = st.selectbox("Chọn cổ phiếu để soi biểu đồ:", filtered_df['Symbol'].tolist())

        if selected_symbol:
            stock_info = filtered_df[filtered_df['Symbol'] == selected_symbol].iloc[0]
            
            # Thẻ tóm tắt chỉ số (KPI Cards)
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Giá hiện tại", f"{stock_info['Close']:.2f}", f"{stock_info['Change_%']:+.2f}%")
            c2.metric("Điểm Kỹ Thuật", f"{stock_info['Score']}/100")
            c3.metric("Cách Đỉnh 52 Tuần", f"-{stock_info['Cách_Đỉnh_%']}%")
            c4.metric("ROE (Quý gần nhất)", f"{stock_info['ROE_%']}%" if use_fa_filter else "N/A")
            c5.metric("Tăng trưởng EPS", f"{stock_info['EPS_Growth_%']}%" if use_fa_filter else "N/A")

            # Vẽ đồ thị nến
            plot_stock_chart(selected_symbol)
    else:
        st.info("💡 Không có mã cổ phiếu nào đạt trọn vẹn tất cả tiêu chí. Bạn hãy thử nới lỏng thông số ở thanh bên trái (Sidebar).")
else:
    st.info("👈 Vui lòng bấm nút **'🚀 BẮT ĐẦU LỌC CỔ PHIẾU'** ở thanh bên trái để quét thị trường.")