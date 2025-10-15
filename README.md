# Hướng dẫn chạy
1. Tạo môi trường ảo và cài đặt các thư viện cần thiết:
   ```bash
   python -m venv fl_env
   source fl_env/bin/activate  # Trên Windows sử dụng `fl_env\Scripts\activate`
   pip install -r requirements.txt
   ```

2. Chạy server:
   ```bash
    python server.py
    ```
3. Chạy các client trong các terminal khác nhau:
   ```bash
    python client.py
    ```