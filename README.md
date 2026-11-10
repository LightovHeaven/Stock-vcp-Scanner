# 📈 Minervini & Livermore Stock Quant Terminal (VN-Stock)

Hệ thống định lượng & lọc cổ phiếu tự động theo phương pháp **Trend Template (Mark Minervini)** và **Pivot Breakout (Jesse Livermore)** tối ưu cho Thị trường Chứng khoán Việt Nam.

## 🚀 Tính năng cốt lõi
1. **Trend Template Filter**: Quét toàn bộ cổ phiếu thỏa mãn tiêu chí MA50 > MA150 > MA200, độ dốc MA200 và khoảng cách đỉnh/đáy 52 tuần.
2. **Breakout & Risk Management**: Nhận diện điểm Pivot phá vỡ kèm khối lượng bùng nổ, tự động tính ngưỡng Cắt lỗ (tối đa 6.5%) và Mục tiêu chốt lời (R:R 1:2.5).
3. **Telegram Real-time Alerts**: Tự động bắn thông báo tức thì khi xuất hiện điểm nổ.
4. **Trading Journal & Position Sizing**: Quản lý rủi ro trên quy mô vốn (quy tắc 1-2% NAV) và tự động thống kê Win-rate, tỷ lệ R:R thực tế qua SQLite.

## 🛠️ Cài đặt & Chạy ứng dụng

```bash
# 1. Clone repository
git clone https://github.com/<your-username>/minervini-livermore-quant.git
cd minervini-livermore-quant

# 2. Cài đặt thư viện
pip install -r requirements.txt

# 3. Chạy ứng dụng Streamlit
streamlit run app.py
