import streamlit as st
import sympy as sp
import numpy as np
import plotly.graph_objects as go
import base64
import os

# ==============================
# CONFIG TRANG (để lên trên cùng)
# ==============================
st.set_page_config(layout="wide", page_title="Khảo sát cực trị")

# ==============================
# LOAD LOGO (an toàn 100%)
# ==============================
def get_base64(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

if os.path.exists("logo.png"):
    logo_base64 = get_base64("logo.png")

    st.markdown(
        f"""
        <style>
        .logo-container {{
            position: fixed;
            bottom: 10px;
            left: 10px;
            z-index: 999;
        }}
        .logo-container img {{
            width: 80px;
            opacity: 0.85;
        }}
        </style>

        <div class="logo-container">
            <img src="data:image/png;base64,{logo_base64}">
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.warning("⚠️ Không tìm thấy file logo.png")

# ==============================
# NỘI DUNG CHÍNH
# ==============================
st.title("🔥 Khảo sát cực trị (Gradient & Hessian)")

st.sidebar.header("🎛 Điều chỉnh tham số")
func_str = st.sidebar.text_input("Nhập hàm f(x,y):", "x^3 + y^3 - 3*x*y")
range_val = st.sidebar.slider("Phạm vi đồ thị", 1, 10, 5)

# Tiền xử lý chuỗi nhập vào (chuyển ^ thành **)
func_str = func_str.replace("^", "**")
x, y = sp.symbols('x y', real=True)

try:
    f = sp.sympify(func_str, locals={'x': x, 'y': y})
except Exception as e:
    st.error(f"❌ Hàm không hợp lệ. Chi tiết lỗi: {e}")
    st.stop()


# TÍNH TOÁN ĐẠO HÀM & HESSIAN
fx = sp.diff(f, x)
fy = sp.diff(f, y)

fxx = sp.diff(fx, x)
fxy = sp.diff(fx, y)
fyx = sp.diff(fy, x)
fyy = sp.diff(fy, y)

H = sp.Matrix([[fxx, fxy], [fyx, fyy]])
detH = H.det()

# Giải hệ phương trình tìm điểm dừng
pts = sp.solve((fx, fy), (x, y), dict=True)
pts_real = []

# Lọc điểm thực một cách an toàn
if isinstance(pts, list):
    for p in pts:
        if x in p and y in p and p[x].is_real and p[y].is_real:
            pts_real.append((p[x], p[y]))


# GIAO DIỆN CHÍNH (CHIA 2 CỘT)

col1, col2 = st.columns([1, 2]) # Cột 1 nhỏ gọn, Cột 2 rộng hơn để vẽ đồ thị

# CỘT 1: HIỂN THỊ CÁC PHÉP TOÁN TOÁN HỌC ---
with col1:
    st.markdown("### 📌 Đạo hàm & Ma trận Hessian")
    st.latex(f"f_x = {sp.latex(fx)}")
    st.latex(f"f_y = {sp.latex(fy)}")
    st.latex(f"H = {sp.latex(H)}")
    st.latex(f"|H| = {sp.latex(detH)}")

    st.markdown("### 📍 Điểm dừng")
    if pts_real:
        for pt in pts_real:
            st.write(f"M( {pt[0]}, {pt[1]} )")
    else:
        st.write("Không có điểm dừng (nghiệm thực).")

    st.markdown("---")
    
    # Tính năng khảo sát điểm bất kỳ
    st.markdown("### 📍 Khảo sát điểm bất kỳ")
    c1, c2 = st.columns(2)
    with c1:
        cx = st.number_input("Tọa độ x =", value=0.0)
    with c2:
        cy = st.number_input("Tọa độ y =", value=0.0)

    # Khôi phục phần tính toán bị thiếu
    grad_x = fx.subs({x: cx, y: cy})
    grad_y = fy.subs({x: cx, y: cy})
    H_val = H.subs({x: cx, y: cy})
    det_val = detH.subs({x: cx, y: cy})

    st.write(f"**Vector Gradient:** ({grad_x}, {grad_y})")
    st.write(f"**|H|:** {det_val}")

    try:
        # Ép kiểu sang số thực để so sánh
        gx_val = float(grad_x)
        gy_val = float(grad_y)
        D = float(det_val)
        A = float(fxx.subs({x: cx, y: cy}))
        
        # BỔ SUNG: Kiểm tra xem có phải là điểm dừng không (xấp xỉ 0 do sai số số học)
        if abs(gx_val) > 1e-6 or abs(gy_val) > 1e-6:
            st.info("👉 Đây không phải là điểm dừng (Vector Gradient khác 0) nên hàm số không đạt cực trị tại đây.")
        else:
            # Nếu đã là điểm dừng thì mới dùng Hessian để phân loại
            if D > 0:
                if A > 0:
                    st.success("👉 Đạt Cực tiểu")
                else:
                    st.success("👉 Đạt Cực đại")
            elif D < 0:
                st.warning("👉 Điểm Yên ngựa")
            else:
                st.info("👉 Không thể kết luận (|H| = 0)")
    except Exception:
        st.error("Không thể phân tích bằng số thực tại điểm này.")
        
#CỘT 2: ĐỒ THỊ MẶT 3D
with col2:
    st.markdown("### 🌐 Đồ thị mặt 3D")
    
    X_vals = np.linspace(-range_val, range_val, 80)
    Y_vals = np.linspace(-range_val, range_val, 80)
    X_grid, Y_grid = np.meshgrid(X_vals, Y_vals)

    f_lam = sp.lambdify((x, y), f, 'numpy')
    
    try:
        Z_grid = f_lam(X_grid, Y_grid)
        # Xử lý trường hợp hàm nhập vào là hằng số
        if np.isscalar(Z_grid):
            Z_grid = np.full_like(X_grid, Z_grid, dtype=float)
    except Exception as e:
        st.error(f"Lỗi khi vẽ đồ thị: {e}")
        st.stop()

    fig = go.Figure()
    fig.add_surface(x=X_grid, y=Y_grid, z=Z_grid, opacity=0.8, colorscale="Viridis")

    # Phân loại và vẽ các điểm dừng lên đồ thị 3D
    for px, py in pts_real:
        try:
            D = float(detH.subs({x: px, y: py}))
            A = float(fxx.subs({x: px, y: py}))
            z_val = float(f.subs({x: px, y: py}))
            px_f, py_f = float(px), float(py)
            
            
            if D > 0:
                if A > 0:
                    label, color = "Cực tiểu", "red"
                else:
                    label, color = "Cực đại", "blue"
            elif D < 0:
                label, color = "Yên ngựa", "black"
            else:
                label, color = "Chưa xác định", "gray"

            fig.add_trace(go.Scatter3d(
                x=[px_f], y=[py_f], z=[z_val],
                mode='markers+text',
                marker=dict(size=8, color=color, symbol='diamond'),
                text=[label],
                textposition="top center",
                name=label
            ))
        except Exception:
            continue

    # Cài đặt giao diện đồ thị (chuyển ngôn ngữ các trục)
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=30),
        height=700,
        scene=dict(
            xaxis_title='Trục X',
            yaxis_title='Trục Y',
            zaxis_title='Trục Z'
        )
    )

    st.plotly_chart(fig, use_container_width=True)
