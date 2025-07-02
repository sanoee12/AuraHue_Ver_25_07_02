import os, cv2, json, math, tempfile, glob
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, jsonify, flash, send_file
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from skimage import color
from scipy.spatial.distance import cdist
import plotly.graph_objs as go
from werkzeug.utils import secure_filename
import time # clear_old_temp_files를 위해 추가
from colormath.color_objects import sRGBColor, CMYKColor # CMYK 개선을 위해 추가
from colormath.color_conversions import convert_color # CMYK 개선을 위해 추가
import re

NCS_HUE_ANGLE = {
    'Y': 0, 'Y10R': 10, 'Y20R': 20, 'Y30R': 30, 'Y40R': 40, 'Y50R': 50,
    'Y60R': 60, 'Y70R': 70, 'Y80R': 80, 'Y90R': 90,
    'R': 100, 'R10B': 110, 'R20B': 120, 'R30B': 130, 'R40B': 140, 'R50B': 150,
    'R60B': 160, 'R70B': 170, 'R80B': 180, 'R90B': 190,
    'B': 200, 'B10G': 210, 'B20G': 220, 'B30G': 230, 'B40G': 240, 'B50G': 250,
    'B60G': 260, 'B70G': 270, 'B80G': 280, 'B90G': 290,
    'G': 300, 'G10Y': 310, 'G20Y': 320, 'G30Y': 330, 'G40Y': 340, 'G50Y': 350
}

app = Flask(__name__)
app.secret_key = 'aura_secret_default'
TEMP_IMAGE_DIR = "static/temp"
os.makedirs(TEMP_IMAGE_DIR, exist_ok=True)

app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True) # Ensure upload folder exists

# Load standard color data
with open('ks_colors.json', encoding='utf-8') as f:
    ks_list = json.load(f)
with open('ncs_colors_with_bw.json', encoding='utf-8') as f:
    ncs_list = json.load(f)
with open('pantone_colors.json', encoding='utf-8') as f:
    pantone_list = json.load(f)

# HUE 각도 변환
def ncs_hue_to_angle(ncs_name):
    # NCS S 1050-Y20R 같은 이름에서 'Y20R' 추출
    hue_match = re.search(r'-([A-Z]+\d*[A-Z]*)$', ncs_name)
    if not hue_match:
        return None
    hue_code = hue_match.group(1)
    # 혹시 사전에 없는 색상코드면 맨 앞/뒤 두 글자만 추출해서라도 매칭
    angle = NCS_HUE_ANGLE.get(hue_code)
    if angle is None and len(hue_code) > 2:
        # 예외적으로 'Y80R'을 'Y80R' 또는 'Y'로 처리
        angle = NCS_HUE_ANGLE.get(hue_code[:1])
    return angle

# Helper functions
def convert_dict(obj):
    if isinstance(obj, dict): return {k: convert_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list): return [convert_dict(x) for x in obj]
    elif isinstance(obj, np.integer): return int(obj)
    elif isinstance(obj, np.floating): return float(obj)
    elif isinstance(obj, np.ndarray): return obj.tolist()
    else: return obj

def resize_image(filepath, max_dim=800):
    img = cv2.imread(filepath)
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
        cv2.imwrite(filepath, img)

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def color_distance(rgb1, rgb2):
    return np.linalg.norm(np.array(rgb1) - np.array(rgb2))

def find_closest_color(target_rgb, std_colors):
    min_dist = float('inf')
    closest = None
    for color_data in std_colors:
        rgb = color_data.get("rgb", hex_to_rgb(color_data["hex"]))
        dist = color_distance(target_rgb, rgb)
        if dist < min_dist:
            min_dist = dist
            closest = color_data
    return closest, min_dist # min_dist도 반환하도록 수정

def white_balance(img):
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:,:,1])
    avg_b = np.average(result[:,:,2])
    # White balance adjustment
    result[:,:,1] = result[:,:,1] - ((avg_a - 128) * (result[:,:,0]/255.0)*1.1)
    result[:,:,2] = result[:,:,2] - ((avg_b - 128) * (result[:,:,0]/255.0)*1.1)
    return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)

def extract_palette(image, n_colors=10):
    pixels = image.reshape(-1, 3)
    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10).fit(pixels)
    palette = np.array(kmeans.cluster_centers_, dtype=np.uint8)
    labels = kmeans.labels_
    counts = np.bincount(labels)
    sorted_idx = np.argsort(counts)[::-1]
    palette = palette[sorted_idx]
    counts = counts[sorted_idx]
    return palette, counts, labels.reshape(image.shape[:2])

def rgb_to_hsv(rgb):
    arr = np.uint8([[rgb]])
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)[0][0]
    return tuple(hsv)

def rgb_to_lab(rgb):
    arr = np.uint8([[rgb]])
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)[0][0]
    return tuple(lab)

def lab_to_lch(lab):
    L, a, b = lab
    C = np.sqrt(a**2 + b**2)
    H = np.degrees(np.arctan2(b, a)) % 360
    return (L, C, H)

def plot_lch_palette(palette, save_path="static/lch3d.html"):
    Ls, Cs, Hs, HEXs = [], [], [], []
    for rgb in palette:
        lab = rgb_to_lab(rgb)
        lch = lab_to_lch(lab)
        Ls.append(lch[0])
        Cs.append(lch[1])
        Hs.append(lch[2])
        HEXs.append('#{:02X}{:02X}{:02X}'.format(*(int(x) for x in rgb)))
    fig = go.Figure(data=[go.Scatter3d(
        x=Ls, y=Cs, z=Hs,
        mode='markers',
        marker=dict(size=10, color=HEXs)
    )])
    fig.write_html(save_path)

# --- 새로 추가된 Plotly 시각화 함수들 ---
def plot_lab_3d(palette, save_path):
    Ls, As, Bs, HEXs = [], [], [], []
    for rgb in palette:
        lab = color.rgb2lab(np.array([[rgb]], dtype=np.uint8)/255.0)[0][0]
        Ls.append(lab[0])
        As.append(lab[1])
        Bs.append(lab[2])
        HEXs.append('#%02x%02x%02x' % tuple(rgb))

    fig = go.Figure(data=[go.Scatter3d(
        x=As, y=Bs, z=Ls,
        mode='markers',
        marker=dict(size=10, color=HEXs, opacity=0.8)
    )])
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=0),
        scene = dict(
            xaxis_title='A (Green-Red)',
            yaxis_title='B (Blue-Yellow)',
            zaxis_title='L (Lightness)'
        )
    )
    fig.write_html(save_path)

def plot_hsv_3d(palette, save_path):
    Hs, Ss, Vs, HEXs = [], [], [], []
    for rgb in palette:
        hsv = cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2HSV)[0][0]
        Hs.append(hsv[0]) # Hue (0-179 in OpenCV)
        Ss.append(hsv[1]) # Saturation (0-255)
        Vs.append(hsv[2]) # Value (0-255)
        HEXs.append('#%02x%02x%02x' % tuple(rgb))

    fig = go.Figure(data=[go.Scatter3d(
        x=Hs, y=Ss, z=Vs,
        mode='markers',
        marker=dict(size=10, color=HEXs, opacity=0.8)
    )])
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=0),
        scene = dict(
            xaxis_title='Hue (0-179)',
            yaxis_title='Saturation (0-255)',
            zaxis_title='Value (0-255)'
        )
    )
    fig.write_html(save_path)

def plot_hsv_2d(palette, save_path):
    Hs, Ss, HEXs = [], [], []
    for rgb in palette:
        hsv = cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2HSV)[0][0]
        Hs.append(hsv[0])
        Ss.append(hsv[1])
        HEXs.append('#%02x%02x%02x' % tuple(rgb))

    fig = go.Figure(data=[go.Scatter(
        x=Hs, y=Ss,
        mode='markers',
        marker=dict(size=10, color=HEXs, opacity=0.8)
    )])
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=0),
        xaxis_title='Hue (0-179)',
        yaxis_title='Saturation (0-255)'
    )
    fig.write_html(save_path)

def plot_lab_2d(palette, save_path):
    As, Bs, HEXs = [], [], []
    for rgb in palette:
        lab = color.rgb2lab(np.array([[rgb]], dtype=np.uint8)/255.0)[0][0]
        As.append(lab[1])
        Bs.append(lab[2])
        HEXs.append('#%02x%02x%02x' % tuple(rgb))

    fig = go.Figure(data=[go.Scatter(
        x=As, y=Bs,
        mode='markers',
        marker=dict(size=10, color=HEXs, opacity=0.8)
    )])
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=0),
        xaxis_title='A (Green-Red)',
        yaxis_title='B (Blue-Yellow)'
    )
    fig.write_html(save_path)

def plot_ncs_3d(palette, ncs_list, save_path):
    Xs, Ys, Zs, HEXs = [], [], [], []
    marker_colors = []
    marker_sizes = []
    hovertexts = []

    for rgb in palette:
        ncs_match, _ = find_closest_color(rgb.tolist(), ncs_list)
        if ncs_match and 'blackness' in ncs_match and 'chromaticness' in ncs_match:
            blackness = ncs_match['blackness']
            chromaticness = ncs_match['chromaticness']

            hue_notation = extract_ncs_hue_notation(ncs_match['name'])
            if hue_notation and hue_notation != 'N':
                angle = ncs_hue_to_angle(hue_notation)
            else:
                angle = 0  # 무채색 or 예외

            Xs.append(blackness)
            Ys.append(chromaticness)
            Zs.append(angle)
            HEXs.append('#%02x%02x%02x' % tuple(rgb))

            # 1) 무채색은 회색/크게, 나머지는 HEX/보통 크기
            if angle == 0:
                marker_colors.append("#888888")
                marker_sizes.append(13)
            else:
                marker_colors.append('#%02x%02x%02x' % tuple(rgb))
                marker_sizes.append(9)

            # 2) hovertext: NCS명, hue, angle 등 원하는 정보
            hovertexts.append(
                f"NCS: {ncs_match['name']}<br>"
                f"Hue: {hue_notation or 'N/A'}<br>"
                f"Angle: {angle}°<br>"
                f"흑:{blackness}, 백:{ncs_match.get('whiteness')}, 순:{chromaticness}"
            )
        else:
            continue

    # 아래 plotly 부분에서 marker/color/size/text 바꿔주기!
    import plotly.graph_objs as go
    fig = go.Figure(data=[go.Scatter3d(
        x=Xs, y=Ys, z=Zs,
        mode='markers',
        marker=dict(
            color=marker_colors,
            size=marker_sizes,
            opacity=0.9
        ),
        text=hovertexts,
        hoverinfo='text'
    )])
    fig.write_html(save_path)

def plot_ncs_2d(palette, ncs_list, save_path):
    xs, ys, hexs = [], [], []
    hovertexts = []

    for rgb in palette:
        ncs_match, _ = find_closest_color(rgb.tolist(), ncs_list)
        if ncs_match and 'blackness' in ncs_match and 'chromaticness' in ncs_match:
            blackness = ncs_match['blackness']
            chromaticness = ncs_match['chromaticness']
            hue_notation = extract_ncs_hue_notation(ncs_match['name'])
            angle = ncs_hue_to_angle(hue_notation) if (hue_notation and hue_notation != 'N') else 0

            xs.append(blackness)  # X축: 흑색도
            ys.append(angle)      # Y축: hue angle로!
            hexs.append('#%02x%02x%02x' % tuple(rgb))

            hovertexts.append(
                f"NCS: {ncs_match['name']}<br>"
                f"Hue: {hue_notation or 'N/A'}<br>"
                f"Angle: {angle}°<br>"
                f"흑:{blackness}, 백:{ncs_match.get('whiteness')}, 순:{chromaticness}"
            )
        else:
            continue

    import plotly.graph_objs as go
    fig = go.Figure(data=[go.Scatter(
        x=xs, y=ys,
        mode='markers',
        marker=dict(color=hexs, size=11, opacity=0.95),
        text=hovertexts,
        hoverinfo='text'
    )])
    fig.update_xaxes(title="흑색도")
    fig.update_yaxes(title="NCS Hue Angle (°)")
    fig.write_html(save_path)
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=0),
        xaxis_title='Hue (0-179)',
        yaxis_title='Saturation (0-255)'
    )
    fig.write_html(save_path)

# 사용자 제공 NCS Hue Angle 변환 함수
def ncs_hue_to_angle(hue_notation):
    # 완전 매칭 우선
    if hue_notation in NCS_HUE_ANGLE:
        return NCS_HUE_ANGLE[hue_notation]
    # 예외적으로 "Y20R" 같은 표기 → 정규식 변환
    m = re.match(r'([A-Z])(\d{2})([A-Z]+)', hue_notation)
    if m:
        base = NCS_HUE_ANGLE.get(m.group(1), 0)
        val = int(m.group(2))
        target = NCS_HUE_ANGLE.get(m.group(3), 0)
        # 선형 보간
        return base + val / 100 * (target - base)
    # 못 찾으면 0
    return 0

def extract_ncs_hue_notation(ncs_name):
    """
    예) 'NCS S 1050-Y20R' → 'Y20R'
        'NCS S 1500-N'   → 'N'
    """
    m = re.search(r'-([A-Z0-9]+)$', ncs_name)
    if m:
        return m.group(1)
    return None

def plot_palette_heatmap(palette, save_path="static/heatmap.png"):
    palette_hsv = cv2.cvtColor(np.uint8([palette]), cv2.COLOR_RGB2HSV)[0]
    h = palette_hsv[:, 0]
    s = palette_hsv[:, 1]
    plt.figure(figsize=(6,2))
    plt.scatter(h, s, c=[color_val/255 for color_val in palette], s=220, marker='s', edgecolors='k')
    plt.xlabel('Hue')
    plt.ylabel('Saturation')
    plt.title("HSV Heatmap")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def get_complementary(rgb):
    comp = [255 - x for x in rgb]
    return '#{:02X}{:02X}{:02X}'.format(*comp)

def group_by_tone(palette):
    tones = {
        'Vivid': [], 'Strong': [], 'Deep': [], 'Bright': [], 'Light': [],
        'Soft': [], 'Light Grayish': [], 'Pale': [],
        'Dull': [], 'Grayish': [], 'Dark Grayish': [], 'Dark': [],
        'White': [], 'Light Grey': [], 'Medium Grey': [], 'Dark Grey': [], 'Black': [],
        'Other': []
    }
    for color_rgb in palette:
        lab = cv2.cvtColor(np.uint8([[color_rgb]]), cv2.COLOR_RGB2LAB)[0][0]
        L, a, b = lab
        C = math.sqrt(a**2 + b**2)
        print(f"L={L:.1f}, C={C:.1f}")
        # ---- 무채색 분리 (Grey/White/Black) ----
        if C < 5:
            if L > 95:
                tones['White'].append(color_rgb.tolist())
            elif L > 80:
                tones['Light Grey'].append(color_rgb.tolist())
            elif L > 40:
                tones['Medium Grey'].append(color_rgb.tolist())
            elif L > 10:
                tones['Dark Grey'].append(color_rgb.tolist())
            else:
                tones['Black'].append(color_rgb.tolist())
            continue

        # ---- PCCS 12톤 분류 ----
        if 60 <= L < 90 and 40 <= C < 90:
            tones['Vivid'].append(color_rgb.tolist())
        elif 50 <= L < 80 and 25 <= C < 65:
            tones['Strong'].append(color_rgb.tolist())
        elif 30 <= L < 60 and 25 <= C < 65:
            tones['Deep'].append(color_rgb.tolist())
        elif 70 <= L < 95 and 20 <= C < 55:
            tones['Bright'].append(color_rgb.tolist())
        elif 75 <= L < 98 and 12 <= C < 30:
            tones['Light'].append(color_rgb.tolist())
        elif 60 <= L < 90 and 10 <= C < 28:
            tones['Soft'].append(color_rgb.tolist())
        elif 45 <= L < 75 and 10 <= C < 28:
            tones['Dull'].append(color_rgb.tolist())
        elif 88 <= L <= 100 and C < 14:
            tones['Pale'].append(color_rgb.tolist())
        elif 72 <= L < 95 and C < 14:
            tones['Light Grayish'].append(color_rgb.tolist())
        elif 35 <= L < 78 and C < 14:
            tones['Grayish'].append(color_rgb.tolist())
        elif 20 <= L < 48 and C < 18:
            tones['Dark Grayish'].append(color_rgb.tolist())
        elif 18 <= L < 50 and 10 <= C < 28:
            tones['Dark'].append(color_rgb.tolist())
        else:
            tones['Other'].append(color_rgb.tolist())
    return tones

def get_palette_recommendations(color_list):
    recs = []
    for i, c_data in enumerate(color_list):
        base = c_data['rgb']
        # Ensure that neighbor1 and neighbor2 are actual hex codes from the palette_colors
        # Use color_list for tone-on-tone and tone-in-tone calculations
        next1 = color_list[(i+1)%len(color_list)]['hex']
        next2 = color_list[(i+2)%len(color_list)]['hex']
        labs = [color.rgb2lab(np.array([[x['rgb']]], dtype=np.uint8)/255.0)[0][0] for x in color_list]
        l_this = color.rgb2lab(np.array([[base]], dtype=np.uint8)/255.0)[0][0][0]

        l_diffs = [abs(l[0] - l_this) for l in labs]
        toneon_idx = np.argsort(l_diffs)[1] # Find the closest L value

        chroma_this = np.sqrt(color.rgb2lab(np.array([[base]], dtype=np.uint8)/255.0)[0][0][1]**2 + color.rgb2lab(np.array([[base]], dtype=np.uint8)/255.0)[0][0][2]**2)
        chroma_diffs = [abs(np.sqrt(l[1]**2 + l[2]**2) - chroma_this) for l in labs]
        tonein_idx = np.argsort(chroma_diffs)[1] # Find the closest chroma value

        recs.append({
            'base': c_data['hex'],
            'complement': c_data['complementary'],
            'neighbor1': next1,
            'neighbor2': next2,
            'toneon': color_list[toneon_idx]['hex'],
            'tonein': color_list[tonein_idx]['hex']
        })
    return recs

def save_palette_excel(color_results, save_path):
    df = pd.DataFrame(color_results)
    df.to_excel(save_path, index=False)

def save_palette_image(palette, save_path):
    img = np.zeros((50, len(palette)*50, 3), np.uint8)
    for idx, rgb in enumerate(palette):
        img[:, idx*50:(idx+1)*50, :] = np.array(rgb, dtype=np.uint8)
    cv2.imwrite(save_path, img)

# WCAG accessibility calculations
def get_luminance(rgb_hex):
    rgb = hex_to_rgb(rgb_hex)
    R = rgb[0] / 255.0
    G = rgb[1] / 255.0
    B = rgb[2] / 255.0

    Rs = R / 12.92 if R <= 0.03928 else ((R + 0.055) / 1.055) ** 2.4
    Gs = G / 12.92 if G <= 0.03928 else ((G + 0.055) / 1.055) ** 2.4
    Bs = B / 12.92 if B <= 0.03928 else ((B + 0.055) / 1.055) ** 2.4

    L = 0.2126 * Rs + 0.7152 * Gs + 0.0722 * Bs
    return L

def get_contrast_ratio(hex1, hex2):
    L1 = get_luminance(hex1)
    L2 = get_luminance(hex2)
    
    if L1 > L2:
        return (L1 + 0.05) / (L2 + 0.05)
    else:
        return (L2 + 0.05) / (L1 + 0.05)

def get_wcag_grade(contrast_ratio):
    if contrast_ratio >= 7.0:
        return "AAA"
    elif contrast_ratio >= 4.5:
        return "AA"
    elif contrast_ratio >= 3.0: # Large text for AA
        return "AA (Large Text)"
    else:
        return "기준미달"

def rgb_to_cmyk_colormath(rgb):
    srgb = sRGBColor(rgb[0]/255.0, rgb[1]/255.0, rgb[2]/255.0)
    cmyk = convert_color(srgb, CMYKColor)
    return [
        round(cmyk.cmyk_c * 100),
        round(cmyk.cmyk_m * 100),
        round(cmyk.cmyk_y * 100),
        round(cmyk.cmyk_k * 100)
    ]

# 임시 파일 정리 함수
def clear_old_temp_files(folder, age_hours=2):
    now = time.time()
    for f in glob.glob(os.path.join(folder, '*')):
        if os.path.isfile(f) and now - os.path.getmtime(f) > age_hours*3600:
            try:
                os.remove(f)
            except Exception:
                pass


# Flask Routes
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    # Get white_balance checkbox value
    apply_white_balance = 'white_balance' in request.form 
    n_colors = int(request.form.get('n_colors', 10))
    if n_colors < 2: n_colors = 2
    if n_colors > 20: n_colors = 20

    if not file:
        flash('파일이 없습니다.', 'error')
        return redirect(url_for('index'))
    filename = secure_filename(file.filename)
    temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(temp_path)
    resize_image(temp_path, max_dim=800)

    img = cv2.imread(temp_path)
    if apply_white_balance: # Apply white balance if checkbox is checked
        img = white_balance(img)
        cv2.imwrite(temp_path, img) # Save the white-balanced image

    return redirect(url_for('analyze_image', filename=filename, n_colors=n_colors)) # Pass n_colors to analyze_image

@app.route('/analyze/<filename>')
def analyze_image(filename):
    # 오래된 임시 파일 정리
    clear_old_temp_files(TEMP_IMAGE_DIR, age_hours=2) #

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image_bgr = cv2.imread(filepath) #

    # 이미지 로드 실패 처리
    if image_bgr is None: #
        flash('이미지 파일을 불러올 수 없습니다. 파일이 손상되었거나 형식이 잘못되었습니다.', 'error') #
        return redirect(url_for('index')) #

    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    n_colors = int(request.args.get('n_colors', 10)) # Get n_colors from URL args
    palette, counts, labelmap = extract_palette(image, n_colors=n_colors)

    # 1. 팔레트 + 정보
    color_list = []
    for idx, color_rgb in enumerate(palette):
        pct = round(int(counts[idx]) / np.sum(counts) * 100, 2)
        lab_val = [float(x) for x in color.rgb2lab(np.array([[color_rgb]], dtype=np.uint8)/255.0)[0][0]]
        
        # Use updated find_closest_color to get delta_e
        ks_match, ks_delta = find_closest_color(color_rgb.tolist(), ks_list)
        ncs_match, ncs_delta = find_closest_color(color_rgb.tolist(), ncs_list)
        pantone_match, pantone_delta = find_closest_color(color_rgb.tolist(), pantone_list)

        # 여기서는 find_closest_color가 None을 반환할 가능성이 낮아 `get("name", "N/A")`를 유지
        ks_name = ks_match.get("name", "N/A") 
        ncs_name = ncs_match.get("name", "N/A")
        pantone_name = pantone_match.get("name", "N/A")
        pantone_delta = pantone_delta or 999  # None 방지
        if pantone_delta > 10:  # ΔE 10 이상이면 유사한 색 없음
            pantone_name = "없음"

        color_list.append({
            'role': '', # 역할은 아직 정해지지 않았으니 일단 비워둠
            'rgb': color_rgb.tolist(),
            'hex': '#{:02X}{:02X}{:02X}'.format(*color_rgb),
            'percentage': pct,
            'complementary': get_complementary(color_rgb.tolist()),
            'lab': lab_val,
            'ks': ks_name,
            'ncs': ncs_name,
            'pantone': pantone_name,
            'ks_hex': ks_match.get("hex", "#FFFFFF") if ks_match else "#FFFFFF",
            'ncs_hex': ncs_match.get("hex", "#FFFFFF") if ncs_match else "#FFFFFF",
            'pantone_hex': pantone_match.get("hex", "#FFFFFF") if pantone_match else "#FFFFFF",
            'pantone_delta_e': pantone_delta, # Add pantone delta E
            'ks_delta_e': ks_delta, # Add KS delta E
            'cmyk': rgb_to_cmyk_colormath(color_rgb.tolist())
        })
    
    # 주조/보조/강조색 역할 부여 (비율 기준)
    if len(color_list) > 0:
        color_list[0]['role'] = '주조색'
    if len(color_list) > 1:
        color_list[1]['role'] = '보조색'
    if len(color_list) > 2:
        color_list[2]['role'] = '강조색'

    # 2. 색상 다양성(표준편차), 대조색
    all_rgbs = np.array([c['rgb'] for c in color_list])
    std_rgb = np.std(all_rgbs, axis=0)
    color_variety = float(np.mean(std_rgb))
    dist_matrix = cdist(all_rgbs, all_rgbs)
    max_idx = np.unravel_index(np.argmax(dist_matrix), dist_matrix.shape)
    most_contrast_pair = (color_list[max_idx[0]], color_list[max_idx[1]])

    # 3. 톤별 묶음 (고급)
    tones = group_by_tone(palette)

    # 4. Heatmap
    temp_heatmap_path = os.path.join(TEMP_IMAGE_DIR, f"heatmap_{filename}.png")
    plot_palette_heatmap(palette, temp_heatmap_path)

    # 5. 배색 추천
    recs = get_palette_recommendations(color_list)

    # 6. Plotly 3D/2D 분포
    temp_lch3d_path = os.path.join(TEMP_IMAGE_DIR, f"lch3d_{filename}.html")
    plot_lch_palette(palette, temp_lch3d_path) # 기존 LCH 3D Plotly

    temp_lab3d_path = os.path.join(TEMP_IMAGE_DIR, f"lab3d_{filename}.html")
    plot_lab_3d(palette, temp_lab3d_path)

    temp_lab2d_path = os.path.join(TEMP_IMAGE_DIR, f"lab2d_{filename}.html")
    plot_lab_2d(palette, temp_lab2d_path)

    temp_hsv3d_path = os.path.join(TEMP_IMAGE_DIR, f"hsv3d_{filename}.html")
    plot_hsv_3d(palette, temp_hsv3d_path)

    temp_hsv2d_path = os.path.join(TEMP_IMAGE_DIR, f"hsv2d_{filename}.html")
    plot_hsv_2d(palette, temp_hsv2d_path)

    temp_ncs3d_path = os.path.join(TEMP_IMAGE_DIR, f"ncs3d_{filename}.html")
    plot_ncs_3d(palette, ncs_list, temp_ncs3d_path)

    temp_ncs2d_path = os.path.join(TEMP_IMAGE_DIR, f"ncs2d_{filename}.html")
    plot_ncs_2d(palette, ncs_list, temp_ncs2d_path)

    # 7. WCAG 접근성 등급 계산
    wcag_grades = []
    if len(color_list) >= 2:
        ratio_p1_p2 = get_contrast_ratio(color_list[0]['hex'], color_list[1]['hex'])
        wcag_grades.append({
            'pair': f"{color_list[0]['hex']} vs {color_list[1]['hex']}",
            'ratio': f"{ratio_p1_p2:.2f}:1",
            'grade': get_wcag_grade(ratio_p1_p2)
        })
    if len(color_list) >= 3:
        ratio_p1_p3 = get_contrast_ratio(color_list[0]['hex'], color_list[2]['hex'])
        wcag_grades.append({
            'pair': f"{color_list[0]['hex']} vs {color_list[2]['hex']}",
            'ratio': f"{ratio_p1_p3:.2f}:1",
            'grade': get_wcag_grade(ratio_p1_p3)
        })
        ratio_p2_p3 = get_contrast_ratio(color_list[1]['hex'], color_list[2]['hex'])
        wcag_grades.append({
            'pair': f"{color_list[1]['hex']} vs {color_list[2]['hex']}",
            'ratio': f"{ratio_p2_p3:.2f}:1",
            'grade': get_wcag_grade(ratio_p2_p3)
        })

    # 8. 상세 정보/다운로드/표 (엑셀)
    excel_path = os.path.join(TEMP_IMAGE_DIR, f"{filename}_palette.xlsx")
    save_palette_excel(color_list, excel_path)

    return render_template(
        'result.html',
        filename=filename,
        original_image=url_for('static', filename=f"uploads/{filename}"),
        palette_colors=convert_dict(color_list),
        heatmap_img=url_for('static', filename=f"temp/heatmap_{filename}.png"),
        tones=convert_dict(tones),
        color_variety=color_variety,
        most_contrast_pair=convert_dict(most_contrast_pair),
        recs=recs,
        table_data=convert_dict(color_list),
        excel_path=url_for('download_excel', filename=filename),
        lab3d_html=url_for('static', filename=f"temp/lab3d_{filename}.html"), # 변경된 경로
        lab2d_html=url_for('static', filename=f"temp/lab2d_{filename}.html"), # 추가된 경로
        hsv3d_html=url_for('static', filename=f"temp/hsv3d_{filename}.html"), # 추가된 경로
        hsv2d_html=url_for('static', filename=f"temp/hsv2d_{filename}.html"), # 추가된 경로
        ncs3d_html=url_for('static', filename=f"temp/ncs3d_{filename}.html"), # 추가된 경로
        ncs2d_html=url_for('static', filename=f"temp/ncs2d_{filename}.html"), # 추가된 경로
        lch3d_html=url_for('static', filename=f"temp/lch3d_{filename}.html"), # LCH 3D도 filename 포함하도록 수정
        wcag_grades=wcag_grades
    )

@app.route('/download_excel/<filename>')
def download_excel(filename):
    excel_path = os.path.join(TEMP_IMAGE_DIR, f"{filename}_palette.xlsx")
    if os.path.exists(excel_path):
        return send_file(excel_path, as_attachment=True)
    else:
        return "파일을 찾을 수 없습니다.", 404

# ---------- 스포이드(이미지 픽셀 정보) API ----------
@app.route('/get_pixel_info', methods=['POST'])
def get_pixel_info():
    data = request.get_json()
    image_path = data.get('image_path')
    x, y = int(data.get('x')), int(data.get('y'))

    if image_path.startswith('/'): image_path = image_path[1:]
    
    # Determine full path based on whether it's a temp or uploaded file
    if image_path.startswith('static/temp'):
        full_image_path = os.path.join(os.getcwd(), image_path)
    elif image_path.startswith('static/uploads'):
        full_image_path = os.path.join(os.getcwd(), image_path)
    else:
        full_image_path = os.path.join(app.config['UPLOAD_FOLDER'], os.path.basename(image_path))

    img = cv2.imread(full_image_path)
    if img is None:
        return jsonify({'error':'이미지를 찾을 수 없습니다.'}), 404

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img_rgb.shape
    if not (0<=x<w and 0<=y<h):
        return jsonify({'error':'좌표범위 에러'}), 400
    pixel = img_rgb[y,x]
    hex_code = '#{:02X}{:02X}{:02X}'.format(*pixel)

    lab_val = [float(val) for val in color.rgb2lab(np.array([[pixel]], dtype=np.uint8)/255.0)[0][0]]
    rgb_pixel = tuple(pixel.tolist())
    
    # Updated find_closest_color returns delta_e
    ks_closest, ks_delta = find_closest_color(rgb_pixel, ks_list)
    ncs_closest, ncs_delta = find_closest_color(rgb_pixel, ncs_list)
    pantone_closest, pantone_delta = find_closest_color(rgb_pixel, pantone_list)

    # CMYK calculation for pixel info (colormath 함수 사용으로 변경 필요)
    # 현재는 기존 수동 계산 유지
    r_norm, g_norm, b_norm = pixel[0]/255.0, pixel[1]/255.0, pixel[2]/255.0
    K = 1 - max(r_norm, g_norm, b_norm)
    if K == 1:
        C, M, Y = 0, 0, 0
    else:
        C = (1 - r_norm - K) / (1 - K)
        M = (1 - g_norm - K) / (1 - K)
        Y = (1 - b_norm - K) / (1 - K)
    cmyk_val = [C * 100, M * 100, Y * 100, K * 100]

    return jsonify({
        'rgb': list(map(int, pixel)),
        'hex': hex_code,
        'cmyk': list(map(float, cmyk_val)),
        'lab': lab_val,
        'ks': ks_closest.get("name", "N/A"),
        'ks_delta_e': float(ks_delta),
        'ncs': ncs_closest.get("name", "N/A"),
        'ncs_hex': ncs_closest.get("name", "N/A"),
        'pantone': pantone_closest.get("name", "N/A"),
        'pantone_delta_e': pantone_delta
    })

# ---------- 크롭(선택영역 분석) ----------
@app.route('/analyze_crop', methods=['POST'])
def analyze_crop():
    data = request.get_json()
    image_path = data.get('image_path')
    x, y, w, h = map(int, [data['x'], data['y'], data['w'], data['h']])

    if image_path.startswith('/'): image_path = image_path[1:]

    if image_path.startswith('static/temp'):
        full_image_path = os.path.join(os.getcwd(), image_path)
    elif image_path.startswith('static/uploads'):
        full_image_path = os.path.join(os.getcwd(), image_path)
    else:
        full_image_path = os.path.join(app.config['UPLOAD_FOLDER'], os.path.basename(image_path))

    img = cv2.imread(full_image_path)
    if img is None:
        return jsonify({'error':'이미지를 찾을 수 없습니다.'}), 404

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    crop_img = img_rgb[y:y+h, x:x+w]
    if crop_img.size == 0:
        return jsonify({'error':'선택된 크롭 영역이 유효하지 않습니다.'}), 400

    palette, counts, _ = extract_palette(crop_img, n_colors=7)

    color_list = []
    for idx, color_rgb in enumerate(palette):
        pct = round(int(counts[idx]) / np.sum(counts) * 100, 2)
        lab_val = [float(x) for x in color.rgb2lab(np.array([[color_rgb]], dtype=np.uint8)/255.0)[0][0]]
        ks_match, _ = find_closest_color(color_rgb.tolist(), ks_list)
        ncs_match, _ = find_closest_color(color_rgb.tolist(), ncs_list)
        pantone_match, pantone_delta = find_closest_color(color_rgb.tolist(), pantone_list)
        
        # CMYK calculation for palette colors (colormath 함수 사용으로 변경 필요)
        # 현재는 기존 수동 계산 유지
        r_norm, g_norm, b_norm = color_rgb[0]/255.0, color_rgb[1]/255.0, color_rgb[2]/255.0
        K = 1 - max(r_norm, g_norm, b_norm)
        if K == 1:
            C, M, Y = 0, 0, 0
        else:
            C = (1 - r_norm - K) / (1 - K)
            M = (1 - g_norm - K) / (1 - K)
            Y = (1 - b_norm - K) / (1 - K)
        cmyk_val = [C * 100, M * 100, Y * 100, K * 100]

        color_list.append({
            'rgb': color_rgb.tolist(),
            'hex': '#{:02X}{:02X}{:02X}'.format(*color_rgb),
            'percentage': pct,
            'complementary': get_complementary(color_rgb.tolist()),
            'lab': lab_val,
            'ks': ks_match.get("name", "N/A"),
            'ncs': ncs_match.get("name", "N/A"),
            'pantone': pantone_match.get("name", "N/A"),
            'pantone_delta_e': pantone_delta,
            'cmyk': list(map(float, cmyk_val))
        })
    return jsonify({'palette': color_list})

if __name__ == '__main__':
    app.run(debug=True)