import cv2
import time
import os
import math
import threading
import random
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
import pygame

MODEL = "hand_landmarker.task"

# SUMBER = "http://192.168.1.11:8080/video"   # ganti IP sesuai yng ada di HP
SUMBER = 0                                # webcam / Iriun / DroidCam dll

LEBAR, TINGGI = 960, 540
FOLDER = "foto"

TAHAN_FOTO = 3.0 
TAHAN_MODE = 2.0
HALUS = 0.40
MIN_BUKA = 120
GOYANG = 9

MULAI_BUTUH = 2       
HENTI_BUTUH = 4       
HALUS_PENA = 0.5      
LOMPAT_MAKS = 160     

AMBANG_JARI = 1.12
AMBANG_JEMPOL = 1.15

# --- ZOOM (pinch satu tangan) ---
ZOOM_MIN, ZOOM_MAX = 1.0, 3.0        # rentang zoom (1x - 3x)
ZOOM_HALUS = 0.15                    # kehalusan transisi zoom
RASIO_PINCH_MIN = 0.35               # jempol & telunjuk rapat -> zoom 1x
RASIO_PINCH_MAX = 1.35               # jempol & telunjuk renggang -> zoom 3x
ZOOM_STABIL_FRAME = 8                # jumlah frame 1-tangan-L berturut sebelum zoom aktif
                                      # (mencegah zoom kepicu saat transisi bentuk bingkai 2 tangan)
ZOOM_PUSAT_HALUS = 0.12              # kehalusan mengejar titik pusat zoom (posisi cubitan)

# --- FOTO EFEK (gestur peace / victory), bisa diganti-ganti ---
TAHAN_MULAI_EFEK = 3.0               # detik gestur ✌ harus DITAHAN dulu sebagai konfirmasi
                                      # (kalau dilepas sebelum genap, batal/reset - anti kepicu tak sengaja)
TAHAN_BLUR = 2.0                     # detik hitung mundur SETELAH konfirmasi berhasil
                                      # (tangan boleh diturunkan, hitungan tetap jalan sendiri)
KUAT_BLUR = 35                       # kekuatan blur gaussian (harus ganjil)

# --- STABILIZER gestur tahan (kepal & peace) ---
TOLERANSI_GESTUR = 6                 # frame kehilangan deteksi yang masih ditoleransi
                                      # (progress ditahan/pause, bukan langsung reset)

# --- MODE BLUR ---
BLUR_KUAT_MIN = 1                    # tidak blur (normal)
BLUR_KUAT_MAX = 75                   # blur maksimal saat peace
BLUR_TRANSISI = 0.12                 # kecepatan transisi blur (0-1)
MUSIK_FOLDER = "musik"
MUSIK_VOLUME = 0.3                   # volume musik (0.0 - 1.0)

# --- THUMBNAIL FOTO TERAKHIR ---
THUMB_W, THUMB_H = 130, 90
THUMB_TAHAN = 4.0                    # detik thumbnail tampil sebelum hilang otomatis

FONT = cv2.FONT_HERSHEY_DUPLEX
CYAN = (255, 235, 0)
MAGENTA = (200, 0, 255)
AMBER = (0, 180, 255)
PUTIH = (240, 245, 250)
ABU = (120, 110, 130)
SAMAR = (55, 50, 62)

os.makedirs(FOLDER, exist_ok=True)

def _lut_retro():
    x = np.arange(256, dtype=np.float32)
    b = np.clip(18 + x * 0.92, 0, 255).astype(np.uint8)
    g = np.clip(6 + x * 0.97, 0, 255).astype(np.uint8)
    r = np.clip(x * 1.06 - 4, 0, 255).astype(np.uint8)
    return cv2.merge([b, g, r]).reshape(1, 256, 3)


LUT_RETRO = _lut_retro()


def _vignette(w, h):
    kx = cv2.getGaussianKernel(w, w * 0.55)
    ky = cv2.getGaussianKernel(h, h * 0.55)
    m = ky @ kx.T
    m = 0.35 + 0.65 * (m / m.max())
    return np.clip(m * 255, 0, 255).astype(np.uint8)[:, :, None].repeat(3, axis=2)


VIGNETTE = _vignette(LEBAR, TINGGI)


def grade_retro(img):
    out = cv2.LUT(img, LUT_RETRO)
    out = cv2.multiply(out, VIGNETTE, scale=1 / 255)
    out[::3] = (out[::3] * 0.78).astype(np.uint8)
    return out


BAYER8 = np.array([
    [0, 32, 8, 40, 2, 34, 10, 42], [48, 16, 56, 24, 50, 18, 58, 26],
    [12, 44, 4, 36, 14, 46, 6, 38], [60, 28, 52, 20, 62, 30, 54, 22],
    [3, 35, 11, 43, 1, 33, 9, 41], [51, 19, 59, 27, 49, 17, 57, 25],
    [15, 47, 7, 39, 13, 45, 5, 37], [63, 31, 55, 23, 61, 29, 53, 21],
], dtype=np.float32) * (255.0 / 64.0)

CLAHE = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
LUT_KONTRAS = np.clip(((np.arange(256) / 255.0) ** 1.6) * 300 - 22,
                      0, 255).astype(np.uint8)
_grain = {}


def abu(roi):
    return cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)


def ke_bgr(g):
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def m_normal(roi, t):
    return roi.copy()


def m_mono(roi, t):
    return ke_bgr(CLAHE.apply(abu(roi)))


def m_kontras(roi, t):
    return ke_bgr(cv2.LUT(CLAHE.apply(abu(roi)), LUT_KONTRAS))


def m_film(roi, t):
    g = CLAHE.apply(abu(roi))
    h, w = g.shape
    if (h, w) not in _grain:
        rng = np.random.default_rng(7)
        _grain[(h, w)] = [rng.normal(0, 11, (h, w)).astype(np.int16)
                          for _ in range(6)]
    g = np.clip(g.astype(np.int16) + _grain[(h, w)][int(t * 18) % 6],
                0, 255).astype(np.uint8)
    bloom = cv2.GaussianBlur(cv2.threshold(g, 195, 255, cv2.THRESH_TOZERO)[1],
                             (0, 0), 6)
    return ke_bgr(cv2.addWeighted(g, 1.0, bloom, 0.35, 0))


def m_garis(roi, t):
    g = cv2.GaussianBlur(abu(roi), (0, 0), 1.2)
    e = cv2.dilate(cv2.Canny(g, 45, 130), np.ones((2, 2), np.uint8))
    return ke_bgr(cv2.add(cv2.multiply(g, 0.14, dtype=cv2.CV_8U), e))


def m_ambang(roi, t):
    g = cv2.GaussianBlur(CLAHE.apply(abu(roi)), (0, 0), 1.0)
    return ke_bgr(cv2.threshold(g, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1])


def m_dither(roi, t):
    g = CLAHE.apply(abu(roi))
    h, w = g.shape
    ubin = np.tile(BAYER8, (h // 8 + 1, w // 8 + 1))[:h, :w]
    return ke_bgr(np.where(g.astype(np.float32) > ubin, 255, 0).astype(np.uint8))


def m_negatif(roi, t):
    return ke_bgr(cv2.bitwise_not(CLAHE.apply(abu(roi))))


LENSA = [("NORMAL", m_normal), ("MONO", m_mono), ("KONTRAS", m_kontras), ("FILM", m_film),
         ("GARIS", m_garis), ("AMBANG", m_ambang), ("DITHER", m_dither),
         ("NEGATIF", m_negatif)]

def urutkan_quad(titik):
    p = np.array(titik, dtype=np.float32)
    c = p.mean(axis=0)
    p = p[np.argsort(np.arctan2(p[:, 1] - c[1], p[:, 0] - c[0]))]
    return np.roll(p, -int(np.argmin(p.sum(axis=1))), axis=0)


def cocokkan(q_baru, q_lama):
    if q_lama is None:
        return q_baru
    terbaik, skor_min = 0, None
    for r in range(4):
        skor = float(np.sum((np.roll(q_baru, -r, axis=0) - q_lama) ** 2))
        if skor_min is None or skor < skor_min:
            skor_min, terbaik = skor, r
    return np.roll(q_baru, -terbaik, axis=0)


def sisi(q):
    return (np.linalg.norm(q[1] - q[0]), np.linalg.norm(q[2] - q[1]),
            np.linalg.norm(q[2] - q[3]), np.linalg.norm(q[3] - q[0]))


def orientasi(q):
    atas, kanan, bawah, kiri = sisi(q)
    v = q[1] - q[0]
    return (math.degrees(math.atan2(v[1], v[0])),
            math.degrees(math.atan2(bawah - atas, bawah + atas + 1e-6)) * 2,
            math.degrees(math.atan2(kanan - kiri, kanan + kiri + 1e-6)) * 2)


def warp_efek(frame, q, fn, t):
    Hf, Wf = frame.shape[:2]
    atas, kanan, bawah, kiri = sisi(q)
    lw, lh = int(max(atas, bawah)), int(max(kiri, kanan))
    if lw < 16 or lh < 16:
        return None, None, None
    tujuan = np.float32([[0, 0], [lw - 1, 0], [lw - 1, lh - 1], [0, lh - 1]])
    rect = cv2.warpPerspective(frame, cv2.getPerspectiveTransform(q, tujuan),
                               (lw, lh))
    efek = fn(rect, t)

    bx1 = max(0, int(np.floor(q[:, 0].min())))
    by1 = max(0, int(np.floor(q[:, 1].min())))
    bx2 = min(Wf, int(np.ceil(q[:, 0].max())) + 1)
    by2 = min(Hf, int(np.ceil(q[:, 1].max())) + 1)
    if bx2 - bx1 < 4 or by2 - by1 < 4:
        return None, None, None

    q_lokal = q - np.float32([bx1, by1])
    balik = cv2.warpPerspective(efek, cv2.getPerspectiveTransform(tujuan, q_lokal),
                                (bx2 - bx1, by2 - by1))
    return efek, balik, (bx1, by1, bx2, by2, q_lokal)


def komposit(tampil, balik, meta):
    bx1, by1, bx2, by2, q_lokal = meta
    mask = np.zeros((by2 - by1, bx2 - bx1), np.uint8)
    cv2.fillConvexPoly(mask, q_lokal.astype(np.int32), 255)
    cv2.copyTo(balik, mask, tampil[by1:by2, bx1:bx2])
    return tampil


def teks(img, s, org, skala, tebal=2, warna=PUTIH, aberasi=True):
    x, y = org
    if aberasi:
        cv2.putText(img, s, (x - 2, y), FONT, skala, MAGENTA, tebal, cv2.LINE_AA)
        cv2.putText(img, s, (x + 2, y), FONT, skala, CYAN, tebal, cv2.LINE_AA)
    cv2.putText(img, s, (x, y), FONT, skala, warna, tebal, cv2.LINE_AA)


def titik_int(p):
    return (int(round(p[0])), int(round(p[1])))


def kurung_quad(img, q, warna, tebal=3):
    for i in range(4):
        p = q[i]
        for j in ((i - 1) % 4, (i + 1) % 4):
            v = q[j] - p
            L = float(np.linalg.norm(v))
            if L < 2:
                continue
            cv2.line(img, titik_int(p), titik_int(p + v / L * min(L * 0.28, 55)),
                     warna, tebal, cv2.LINE_AA)


def cincin(img, pusat, radius, maju, warna, tebal=4):
    cv2.ellipse(img, pusat, (radius, radius), -90, 0, 360, SAMAR, tebal)
    if maju > 0:
        cv2.ellipse(img, pusat, (radius, radius), -90, 0,
                    int(360 * min(1.0, maju)), warna, tebal, cv2.LINE_AA)


def jarak(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def jari_terbuka(hand):
    """[jempol, telunjuk, tengah, manis, kelingking]"""
    w, ref = hand[0], hand[17]
    out = [jarak(hand[4], ref) > jarak(hand[2], ref) * AMBANG_JEMPOL]
    for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
        out.append(jarak(hand[tip], w) > jarak(hand[pip], w) * AMBANG_JARI)
    return out


def is_L(f):
    return f[0] and f[1] and not f[2] and not f[3] and not f[4]


def is_tunjuk(f):
    """Menunjuk - jempol DIABAIKAN. Dipakai saat menggambar."""
    return f[1] and not f[2] and not f[3] and not f[4]


def is_tunjuk_ketat(f):
    """Menunjuk dengan jempol ditekuk. Dipakai agar tidak bentrok dengan L."""
    return is_tunjuk(f) and not f[0]


def is_peace(f):
    return f[1] and f[2] and not f[3] and not f[4]


def is_tiga_jari(f):
    """Telunjuk + tengah + manis terbuka, jempol & kelingking tertutup."""
    return not f[0] and f[1] and f[2] and f[3] and not f[4]


def is_kepal(f):
    return not any(f)


def is_telapak(f):
    return all(f)


def rasio_pinch(hand, w, h):
    """Jarak jempol-telunjuk dinormalisasi ukuran tangan (skala-independen)."""
    thumb = np.array([hand[4].x * w, hand[4].y * h])
    index = np.array([hand[8].x * w, hand[8].y * h])
    wrist = np.array([hand[0].x * w, hand[0].y * h])
    mcp = np.array([hand[9].x * w, hand[9].y * h])
    ukuran = np.linalg.norm(mcp - wrist) + 1e-6
    return float(np.linalg.norm(thumb - index) / ukuran)


def terapkan_zoom(img, zoom, pusat=None):
    """Digital zoom: crop di sekitar 'pusat' (posisi tangan/cubitan) lalu
    skalakan balik ke ukuran asli. Kalau pusat None, crop di tengah."""
    if zoom <= 1.01:
        return img
    h, w = img.shape[:2]
    nw, nh = int(w / zoom), int(h / zoom)
    if pusat is None:
        cx, cy = w / 2, h / 2
    else:
        cx, cy = pusat
    x1 = int(cx - nw / 2)
    y1 = int(cy - nh / 2)
    x1 = max(0, min(w - nw, x1))
    y1 = max(0, min(h - nh, y1))
    crop = img[y1:y1 + nh, x1:x1 + nw]
    return cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)


def buat_blur(img, kuat=KUAT_BLUR):
    k = kuat | 1  # pastikan ganjil
    return cv2.GaussianBlur(img, (k, k), 0)


_MATRIKS_SEPIA = np.array([[0.272, 0.534, 0.131],
                           [0.349, 0.686, 0.168],
                           [0.393, 0.769, 0.189]])


def buat_normal(img):
    return img.copy()


def buat_sepia(img):
    sep = cv2.transform(img, _MATRIKS_SEPIA)
    return np.clip(sep, 0, 255).astype(np.uint8)


def buat_sketsa(img):
    abu_sketsa, _ = cv2.pencilSketch(img, sigma_s=60, sigma_r=0.07, shade_factor=0.05)
    return ke_bgr(abu_sketsa)


def buat_glow(img):
    lembut = cv2.GaussianBlur(img, (0, 0), 15)
    return cv2.addWeighted(img, 0.75, lembut, 0.55, 8)


def buat_cermin(img):
    h, w = img.shape[:2]
    nw = w // 2
    kiri = img[:, :nw]
    gabung = np.hstack([kiri, cv2.flip(kiri, 1)])
    if gabung.shape[1] != w:
        gabung = cv2.resize(gabung, (w, h))
    return gabung


EFEK_PEACE = [("NORMAL", buat_normal), ("BLUR", buat_blur), ("SEPIA", buat_sepia), ("SKETSA", buat_sketsa),
             ("GLOW", buat_glow), ("CERMIN", buat_cermin)]


# ====================== BINGKAI FOTO LUCU (TEMA) ======================
# Semua hiasan digambar sendiri pakai OpenCV (bukan tempelan gambar dari luar).
# Tiap tema = warna latar padding + garis pinggir + daftar "stiker" doodle yang
# ditaburkan padat di area padding (bukan cuma di tepi tipis seperti sebelumnya).
# Opsional: dibuatkan juga versi .gif berkedip-kedip (butuh Pillow, best-effort).

def _titik_bintang(cx, cy, r_luar, r_dalam, putar=-90):
    pts = []
    for i in range(10):
        r = r_luar if i % 2 == 0 else r_dalam
        a = math.radians(putar + i * 36)
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return np.array(pts, np.int32)


def gambar_bintang(img, c, r=14, warna=AMBER):
    pts = _titik_bintang(c[0], c[1], r, r * 0.45)
    cv2.fillPoly(img, [pts], warna, cv2.LINE_AA)
    cv2.polylines(img, [pts], True, PUTIH, 1, cv2.LINE_AA)


def gambar_hati(img, c, r=12, warna=MAGENTA):
    cx, cy = c
    cv2.circle(img, (cx - r // 2, cy - r // 3), max(2, r // 2), warna, cv2.FILLED, cv2.LINE_AA)
    cv2.circle(img, (cx + r // 2, cy - r // 3), max(2, r // 2), warna, cv2.FILLED, cv2.LINE_AA)
    segi = np.array([[cx - r, cy - r // 4], [cx + r, cy - r // 4], [cx, cy + r]], np.int32)
    cv2.fillPoly(img, [segi], warna, cv2.LINE_AA)


def gambar_bulan(img, c, r=12, warna=(230, 230, 255)):
    """Bulan sabit: gambar lingkaran penuh lalu 'gigit' pakai warna latar sekitar."""
    cx, cy = c
    h, w = img.shape[:2]
    bx = min(w - 1, max(0, cx + int(r * 0.45)))
    by = min(h - 1, max(0, cy - int(r * 0.1)))
    warna_bg = tuple(int(v) for v in img[by, bx])
    cv2.circle(img, (cx, cy), r, warna, cv2.FILLED, cv2.LINE_AA)
    cv2.circle(img, (cx + int(r * 0.45), cy - int(r * 0.1)), int(r * 0.85),
               warna_bg, cv2.FILLED, cv2.LINE_AA)


def gambar_bunga(img, c, r=12, warna=(255, 180, 210)):
    cx, cy = c
    kelopak = max(2, int(r * 0.55))
    for i in range(5):
        a = math.radians(i * 72)
        px, py = int(cx + r * 0.55 * math.cos(a)), int(cy + r * 0.55 * math.sin(a))
        cv2.circle(img, (px, py), kelopak, warna, cv2.FILLED, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), max(2, int(r * 0.4)), AMBER, cv2.FILLED, cv2.LINE_AA)


def gambar_awan(img, c, r=12, warna=PUTIH):
    cx, cy = c
    for dx, dy, rr in [(-r * 0.5, 0, r * 0.55), (0, -r * 0.25, r * 0.7), (r * 0.5, 0, r * 0.55)]:
        cv2.circle(img, (int(cx + dx), int(cy + dy)), max(2, int(rr)), warna, cv2.FILLED, cv2.LINE_AA)
    cv2.ellipse(img, (cx, int(cy + r * 0.25)), (int(r * 0.9), int(r * 0.35)), 0, 0, 360,
                warna, cv2.FILLED, cv2.LINE_AA)


def gambar_kilau(img, c, r=10, warna=PUTIH):
    cx, cy = c
    cv2.line(img, (cx - r, cy), (cx + r, cy), warna, 2, cv2.LINE_AA)
    cv2.line(img, (cx, cy - r), (cx, cy + r), warna, 2, cv2.LINE_AA)
    d = int(r * 0.6)
    cv2.line(img, (cx - d, cy - d), (cx + d, cy + d), warna, 1, cv2.LINE_AA)
    cv2.line(img, (cx - d, cy + d), (cx + d, cy - d), warna, 1, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), max(1, int(r * 0.2)), warna, cv2.FILLED, cv2.LINE_AA)


def gambar_kupu(img, c, r=12, warna=(150, 220, 255)):
    cx, cy = c
    cv2.ellipse(img, (cx - int(r * 0.4), cy), (int(r * 0.5), int(r * 0.7)), 20, 0, 360,
                warna, cv2.FILLED, cv2.LINE_AA)
    cv2.ellipse(img, (cx + int(r * 0.4), cy), (int(r * 0.5), int(r * 0.7)), -20, 0, 360,
                warna, cv2.FILLED, cv2.LINE_AA)
    cv2.line(img, (cx, cy - int(r * 0.6)), (cx, cy + int(r * 0.6)), (60, 50, 50), 2, cv2.LINE_AA)


def gambar_confetti_kecil(img, c, r=8, warna=CYAN):
    cx, cy = c
    if random.random() < 0.5:
        cv2.circle(img, c, max(2, int(r * 0.5)), warna, cv2.FILLED, cv2.LINE_AA)
    else:
        s = max(2, int(r * 0.5))
        rect = cv2.boxPoints(((cx, cy), (s * 2, s * 2), random.randint(0, 360))).astype(np.int32)
        cv2.fillConvexPoly(img, rect, warna)


def gambar_jaring(img, c, r=12, warna=PUTIH):
    """Doodle pola jaring generik (garis + lingkaran konsentris) - bukan logo siapa pun."""
    cx, cy = c
    for i in range(6):
        a = math.radians(i * 60)
        x2 = int(cx + r * math.cos(a))
        y2 = int(cy + r * math.sin(a))
        cv2.line(img, (cx, cy), (x2, y2), warna, 1, cv2.LINE_AA)
    for rr in (r * 0.4, r * 0.7, r):
        cv2.circle(img, (cx, cy), max(1, int(rr)), warna, 1, cv2.LINE_AA)


def gambar_siluet_malam(img, c, r=12, warna=(20, 20, 20)):
    """Siluet generik bersayap ala malam hari - doodle orisinal, bukan logo merek apa pun."""
    cx, cy = c
    pts = np.array([
        (cx - 2 * r, cy), (cx - int(r * 0.6), cy - int(r * 0.3)),
        (cx - int(r * 0.3), cy - int(r * 0.1)), (cx, cy - int(r * 0.5)),
        (cx + int(r * 0.3), cy - int(r * 0.1)), (cx + int(r * 0.6), cy - int(r * 0.3)),
        (cx + 2 * r, cy), (cx + int(r * 0.5), cy + int(r * 0.15)),
        (cx, cy + int(r * 0.05)), (cx - int(r * 0.5), cy + int(r * 0.15)),
    ], np.int32)
    cv2.fillPoly(img, [pts], warna, cv2.LINE_AA)


def gambar_kilat(img, c, r=12, warna=AMBER):
    cx, cy = c
    pts = np.array([
        (cx, cy - r), (cx - int(r * 0.3), cy),
        (cx + int(r * 0.1), cy), (cx - int(r * 0.15), cy + r),
        (cx + int(r * 0.35), cy + int(r * 0.15)), (cx - int(r * 0.05), cy + int(r * 0.15)),
    ], np.int32)
    cv2.fillPoly(img, [pts], warna, cv2.LINE_AA)


def gambar_perisai(img, c, r=12, warna=CYAN):
    cx, cy = c
    pts = np.array([
        (cx - r, cy - int(r * 0.6)), (cx + r, cy - int(r * 0.6)),
        (cx + r, cy + int(r * 0.1)), (cx, cy + r), (cx - r, cy + int(r * 0.1)),
    ], np.int32)
    cv2.fillPoly(img, [pts], warna, cv2.LINE_AA)
    cv2.polylines(img, [pts], True, PUTIH, 1, cv2.LINE_AA)


def _gambar_chibi_dasar(img, c, r, warna_suit, emblem=None, jubah=False):
    """Chibi pahlawan generik: kepala bulat + badan + pose kepal tangan.
    Desain orisinal (mata bulat sederhana, bukan lensa segitiga khas karakter
    tertentu) - hanya nuansa 'pahlawan', bukan tiruan karakter berhak cipta."""
    cx, cy = c
    r = max(7, r)
    aksen = PUTIH if (0.114 * warna_suit[0] + 0.587 * warna_suit[1] + 0.299 * warna_suit[2]) < 150 else (30, 28, 32)

    if jubah:
        pts = np.array([
            (cx - int(r * 0.55), cy - int(r * 0.1)),
            (cx - int(r * 1.15), cy + int(r * 1.3)),
            (cx + int(r * 0.1), cy + int(r * 0.9)),
        ], np.int32)
        cv2.fillPoly(img, [pts], warna_suit, cv2.LINE_AA)

    # badan (oval)
    cv2.ellipse(img, (cx, cy + int(r * 0.55)), (int(r * 0.68), int(r * 0.62)), 0, 0, 360,
                warna_suit, cv2.FILLED, cv2.LINE_AA)
    # lengan pose kepal ke atas
    tebal = max(1, int(r * 0.22))
    cv2.line(img, (cx - int(r * 0.55), cy + int(r * 0.35)),
             (cx - int(r * 1.05), cy - int(r * 0.05)), warna_suit, tebal, cv2.LINE_AA)
    cv2.line(img, (cx + int(r * 0.55), cy + int(r * 0.35)),
             (cx + int(r * 1.05), cy - int(r * 0.05)), warna_suit, tebal, cv2.LINE_AA)
    cv2.circle(img, (cx - int(r * 1.05), cy - int(r * 0.05)), max(1, int(r * 0.16)), warna_suit, cv2.FILLED, cv2.LINE_AA)
    cv2.circle(img, (cx + int(r * 1.05), cy - int(r * 0.05)), max(1, int(r * 0.16)), warna_suit, cv2.FILLED, cv2.LINE_AA)
    # kepala
    cv2.circle(img, (cx, cy - int(r * 0.35)), int(r * 0.58), warna_suit, cv2.FILLED, cv2.LINE_AA)
    # mata bulat sederhana (BUKAN lensa segitiga)
    eye_r = max(1, int(r * 0.14))
    ex = int(r * 0.22)
    cv2.circle(img, (cx - ex, cy - int(r * 0.4)), eye_r, aksen, cv2.FILLED, cv2.LINE_AA)
    cv2.circle(img, (cx + ex, cy - int(r * 0.4)), eye_r, aksen, cv2.FILLED, cv2.LINE_AA)

    # emblem kecil di dada (bentuk generik, bukan logo siapa pun)
    ec = (cx, cy + int(r * 0.5))
    if emblem == "bintang":
        gambar_bintang(img, ec, max(2, int(r * 0.3)), aksen)
    elif emblem == "kilat":
        gambar_kilat(img, ec, max(2, int(r * 0.32)), aksen)
    elif emblem == "perisai":
        gambar_perisai(img, ec, max(2, int(r * 0.3)), aksen)
    elif emblem == "lingkaran":
        cv2.circle(img, ec, max(2, int(r * 0.16)), aksen, 1, cv2.LINE_AA)
        cv2.circle(img, ec, max(1, int(r * 0.08)), aksen, cv2.FILLED, cv2.LINE_AA)


def gambar_chibi_jaring(img, c, r, warna):
    _gambar_chibi_dasar(img, c, r, warna, emblem="lingkaran")


def gambar_chibi_malam(img, c, r, warna):
    _gambar_chibi_dasar(img, c, r, warna, jubah=True)


def gambar_chibi_kilat(img, c, r, warna):
    _gambar_chibi_dasar(img, c, r, warna, emblem="kilat")


def gambar_chibi_perisai(img, c, r, warna):
    _gambar_chibi_dasar(img, c, r, warna, emblem="perisai")


def gambar_chibi_bintang(img, c, r, warna):
    _gambar_chibi_dasar(img, c, r, warna, emblem="bintang", jubah=True)


def gambar_piringan_hitam(img, c, r=14, warna=PUTIH):
    cx, cy = c
    cv2.circle(img, (cx, cy), r, (15, 15, 18), cv2.FILLED, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), max(2, int(r * 0.7)), (35, 35, 40), 1, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), max(2, int(r * 0.45)), warna, cv2.FILLED, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), max(1, int(r * 0.15)), (15, 15, 18), cv2.FILLED, cv2.LINE_AA)


def gambar_not_musik(img, c, r=12, warna=MAGENTA):
    cx, cy = c
    r = max(6, r)
    cv2.circle(img, (cx - r // 3, cy + r // 3), max(2, r // 3), warna, cv2.FILLED, cv2.LINE_AA)
    cv2.circle(img, (cx + r // 3, cy + r // 6), max(2, r // 3), warna, cv2.FILLED, cv2.LINE_AA)
    cv2.line(img, (cx - r // 3 + max(2, r // 3), cy + r // 3), (cx - r // 3 + max(2, r // 3), cy - r // 2), warna, 2, cv2.LINE_AA)
    cv2.line(img, (cx + r // 3 + max(2, r // 3), cy + r // 6), (cx + r // 3 + max(2, r // 3), cy - int(r // 1.5)), warna, 2, cv2.LINE_AA)
    cv2.line(img, (cx - r // 3 + max(2, r // 3), cy - r // 2), (cx + r // 3 + max(2, r // 3), cy - int(r // 1.5)), warna, 2, cv2.LINE_AA)


def gambar_kaset(img, c, r=14, warna=AMBER):
    cx, cy = c
    w_kaset, h_kaset = r * 2, int(r * 1.3)
    cv2.rectangle(img, (cx - w_kaset // 2, cy - h_kaset // 2), (cx + w_kaset // 2, cy + h_kaset // 2), warna, 1, cv2.LINE_AA)
    cv2.circle(img, (cx - r // 2, cy), max(2, r // 4), warna, 1, cv2.LINE_AA)
    cv2.circle(img, (cx + r // 2, cy), max(2, r // 4), warna, 1, cv2.LINE_AA)


TEMA_ABOUT_YOU = dict(bg=(22, 22, 28), border=(210, 200, 225),
                       stiker=[gambar_piringan_hitam, gambar_not_musik, gambar_kilau],
                       warna=[(220, 215, 230), (160, 150, 180), AMBER],
                       tape=False, pad=0.13, judul="ABOUT YOU - THE 1975",
                       tipe="musik", lagu="About You", artis="The 1975",
                       lirik="\"Do you think I have forgotten about you?\"",
                       durasi="5:26", posisi="2:14", kateg="LAGU")

TEMA_GLIMPSE_OF_US = dict(bg=(26, 20, 32), border=(190, 160, 220),
                          stiker=[gambar_piringan_hitam, gambar_hati, gambar_bulan],
                          warna=[(210, 180, 240), (160, 140, 200), PUTIH],
                          tape=False, pad=0.13, judul="GLIMPSE OF US - JOJI",
                          tipe="musik", lagu="Glimpse of Us", artis="Joji",
                          lirik="\"'Cause sometimes I look in her eyes & see a glimpse of us\"",
                          durasi="3:53", posisi="1:48", kateg="LAGU")

TEMA_YELLOW = dict(bg=(255, 248, 220), border=(0, 185, 245),
                    stiker=[gambar_bintang, gambar_kilau, gambar_not_musik],
                    warna=[AMBER, (0, 200, 255), (100, 100, 100)],
                    tape=False, pad=0.13, judul="YELLOW - COLDPLAY",
                    tipe="musik", lagu="Yellow", artis="Coldplay",
                    lirik="\"Look at the stars, look how they shine for you\"",
                    durasi="4:29", posisi="2:05", kateg="LAGU")

TEMA_NOTHIN_ON_YOU = dict(bg=(255, 240, 245), border=(255, 100, 150),
                          stiker=[gambar_hati, gambar_not_musik, gambar_kilau],
                          warna=[(255, 100, 150), AMBER, PUTIH],
                          tape=False, pad=0.14, judul="NOTHIN' ON YOU - B.O.B & BRUNO MARS",
                          tipe="musik", lagu="Nothin' On You", artis="B.o.B ft. Bruno Mars",
                          lirik="\"Nothin' on you, baby, Nothin' on you\"",
                          durasi="4:27", posisi="1:50", kateg="LAGU")

TEMA_DIE_WITH_A_SMILE = dict(bg=(20, 18, 28), border=(220, 180, 255),
                             stiker=[gambar_piringan_hitam, gambar_hati, gambar_kilau],
                             warna=[(220, 180, 255), CYAN, PUTIH],
                             tape=False, pad=0.14, judul="DIE WITH A SMILE - LADY GAGA & BRUNO MARS",
                             tipe="musik", lagu="Die With A Smile", artis="Lady Gaga & Bruno Mars",
                             lirik="\"If the world was ending, I'd wanna be next to you\"",
                             durasi="4:11", posisi="2:30", kateg="LAGU")

TEMA_PERFECT = dict(bg=(252, 246, 232), border=(240, 170, 70),
                    stiker=[gambar_bintang, gambar_hati, gambar_bunga],
                    warna=[(240, 170, 70), MAGENTA, PUTIH],
                    tape=False, pad=0.14, judul="PERFECT - ED SHEERAN",
                    tipe="musik", lagu="Perfect", artis="Ed Sheeran",
                    lirik="\"Darling, you look perfect tonight\"",
                    durasi="4:23", posisi="2:10", kateg="LAGU")

TEMA_UNTIL_I_FOUND_YOU = dict(bg=(24, 22, 34), border=(170, 195, 255),
                              stiker=[gambar_piringan_hitam, gambar_not_musik, gambar_bulan],
                              warna=[(170, 195, 255), CYAN, AMBER],
                              tape=False, pad=0.14, judul="UNTIL I FOUND YOU - STEPHEN SANCHEZ",
                              tipe="musik", lagu="Until I Found You", artis="Stephen Sanchez",
                              lirik="\"I would never fall in love until I found her\"",
                              durasi="2:57", posisi="1:15", kateg="LAGU")

TEMA_GOLDEN_HOUR = dict(bg=(255, 244, 210), border=(255, 160, 0),
                        stiker=[gambar_bintang, gambar_kilau, gambar_not_musik],
                        warna=[(255, 160, 0), AMBER, PUTIH],
                        tape=False, pad=0.14, judul="GOLDEN HOUR - JVKE",
                        tipe="musik", lagu="Golden Hour", artis="JVKE",
                        lirik="\"Shine your light on me, 'cause I don't wanna sleep\"",
                        durasi="3:29", posisi="1:40", kateg="LAGU")

TEMA_LINE_WITHOUT_HOOK = dict(bg=(235, 242, 255), border=(70, 135, 240),
                               stiker=[gambar_not_musik, gambar_awan, gambar_bunga],
                               warna=[(70, 135, 240), MAGENTA, AMBER],
                               tape=False, pad=0.14, judul="LINE WITHOUT A HOOK - RICKY MONTGOMERY",
                               tipe="musik", lagu="Line Without a Hook", artis="Ricky Montgomery",
                               lirik="\"She's a lady, and I'm just a line without a hook\"",
                               durasi="4:10", posisi="2:02", kateg="LAGU")

TEMA_VINTAGE_CASSETTE = dict(bg=(32, 28, 30), border=AMBER,
                            stiker=[gambar_kaset, gambar_not_musik, gambar_kilau],
                            warna=[AMBER, CYAN, PUTIH],
                            tape=True, tape_warna=[AMBER, (0, 140, 255)], pad=0.12,
                            judul="MIXTAPE 90s - CASSETTE", tipe="kaset", kateg="VINTAGE")

TEMA_FILM_STRIP = dict(bg=(16, 16, 16), border=PUTIH,
                        stiker=[gambar_kilau],
                        warna=[PUTIH, ABU],
                        tape=False, pad=0.13, judul="35MM CINEMA FILM STRIP",
                        tipe="film_strip", kateg="VINTAGE")

TEMA_MAGAZINE = dict(bg=(250, 248, 245), border=(30, 30, 30),
                      stiker=[gambar_bintang, gambar_kilau],
                      warna=[(30, 30, 30), MAGENTA, AMBER],
                      tape=False, pad=0.13, judul="RETRO VOGUE EDITORIAL",
                      tipe="magazine", kateg="VINTAGE")

TEMA_CYBERPUNK = dict(bg=(12, 10, 24), border=CYAN,
                       stiker=[gambar_kilat, gambar_kilau, gambar_jaring],
                       warna=[CYAN, MAGENTA, AMBER],
                       tape=False, pad=0.12, judul="CYBERPUNK 2077 HUD",
                       tipe="cyber", kateg="VINTAGE")

TEMA_POLAROID = dict(bg=(250, 250, 250), border=(225, 220, 215),
                      stiker=[gambar_hati, gambar_kilau],
                      warna=[MAGENTA, AMBER, (255, 200, 220)],
                      tape=False, pad=0.11, judul="RETROLENS", kateg="VINTAGE")

TEMA_BINTANG = dict(bg=(24, 22, 30), border=AMBER,
                     stiker=[gambar_bintang, gambar_kilau, gambar_bulan],
                     warna=[AMBER, CYAN, PUTIH],
                     tape=False, pad=0.11, judul="STARLIGHT", kateg="LUCU")

TEMA_HATI = dict(bg=(250, 232, 242), border=MAGENTA,
                  stiker=[gambar_hati, gambar_bunga],
                  warna=[MAGENTA, (170, 120, 255), (255, 150, 200)],
                  tape=False, pad=0.11, judul="LOVE", kateg="LUCU")

TEMA_WASHI = dict(bg=(245, 245, 240), border=(200, 200, 195),
                   stiker=[gambar_bunga, gambar_awan],
                   warna=[(180, 220, 255), (255, 210, 230), (210, 255, 210)],
                   tape=True, tape_warna=[(180, 220, 255), (255, 210, 230), (210, 255, 210), (255, 235, 180)],
                   pad=0.12, judul="SCRAPBOOK", kateg="LUCU")

TEMA_CONFETTI = dict(bg=(28, 26, 34), border=CYAN,
                      stiker=[gambar_confetti_kecil, gambar_kilau, gambar_bintang],
                      warna=[CYAN, MAGENTA, AMBER, PUTIH, (0, 255, 150)],
                      tape=False, pad=0.11, judul="PARTY", kateg="LUCU")

TEMA_GALAKSI = dict(bg=(30, 16, 48), border=(255, 170, 60),
                     stiker=[gambar_bulan, gambar_bintang, gambar_kilau],
                     warna=[PUTIH, (200, 190, 255), (255, 210, 120)],
                     tape=False, pad=0.11, judul="GALAXY", kateg="LUCU")

TEMA_KEBUN = dict(bg=(233, 250, 235), border=(90, 180, 110),
                   stiker=[gambar_bunga, gambar_awan, gambar_kupu],
                   warna=[(255, 180, 210), (255, 255, 255), (150, 220, 255)],
                   tape=False, pad=0.11, judul="GARDEN", kateg="LUCU")

# --- Tema bergaya pahlawan super (orisinal, bukan karakter/logo berhak cipta) ---
TEMA_JARING_MERAH = dict(bg=(235, 235, 245), border=(30, 40, 180),
                          stiker=[gambar_chibi_jaring, gambar_jaring, gambar_kilau],
                          warna=[(30, 30, 200), (0, 0, 255), PUTIH],
                          tape=False, pad=0.11, judul="WEB HERO", kateg="HERO")

TEMA_MALAM_KOTA = dict(bg=(18, 18, 22), border=(230, 190, 40),
                        stiker=[gambar_chibi_malam, gambar_siluet_malam, gambar_bintang],
                        warna=[(230, 190, 40), PUTIH, (20, 20, 20)],
                        tape=False, pad=0.11, judul="NIGHT GUARDIAN", kateg="HERO")

TEMA_KILAT_BIRU = dict(bg=(255, 244, 214), border=(0, 80, 220),
                        stiker=[gambar_chibi_kilat, gambar_kilat, gambar_bintang],
                        warna=[(0, 120, 255), (0, 0, 200), AMBER],
                        tape=False, pad=0.11, judul="SPEED HERO", kateg="HERO")

TEMA_PERISAI_MERAH = dict(bg=(230, 230, 255), border=(0, 0, 180),
                           stiker=[gambar_chibi_perisai, gambar_perisai, gambar_bintang],
                           warna=[(0, 0, 200), (0, 0, 255), PUTIH],
                           tape=False, pad=0.11, judul="SHIELD HERO", kateg="HERO")

TEMA_AMAZON = dict(bg=(255, 245, 225), border=(0, 140, 190),
                    stiker=[gambar_chibi_bintang, gambar_bintang, gambar_kilau],
                    warna=[(0, 140, 190), AMBER, (0, 0, 200)],
                    tape=False, pad=0.11, judul="WARRIOR HERO", kateg="HERO")

TEMA_LOVER = None
TEMA_AS_IT_WAS = None
BINGKAI_LUCU = [
    ("ABOUT YOU", TEMA_ABOUT_YOU),
    ("GLIMPSE OF US", TEMA_GLIMPSE_OF_US),
    ("YELLOW", TEMA_YELLOW),
    ("AS IT WAS", TEMA_AS_IT_WAS),
    ("LOVER", TEMA_LOVER),
    ("NOTHIN' ON YOU", TEMA_NOTHIN_ON_YOU),
    ("DIE WITH A SMILE", TEMA_DIE_WITH_A_SMILE),
    ("PERFECT", TEMA_PERFECT),
    ("UNTIL I FOUND YOU", TEMA_UNTIL_I_FOUND_YOU),
    ("GOLDEN HOUR", TEMA_GOLDEN_HOUR),
    ("LINE WITHOUT A HOOK", TEMA_LINE_WITHOUT_HOOK),
    ("CASSETTE 90S", TEMA_VINTAGE_CASSETTE),
    ("FILM STRIP 35MM", TEMA_FILM_STRIP),
    ("VOGUE MAG", TEMA_MAGAZINE),
    ("CYBERPUNK", TEMA_CYBERPUNK),
    ("POLAROID", TEMA_POLAROID),
    ("BINTANG", TEMA_BINTANG),
    ("HATI", TEMA_HATI),
    ("WASHI", TEMA_WASHI),
    ("CONFETTI", TEMA_CONFETTI),
    ("GALAKSI", TEMA_GALAKSI),
    ("KEBUN", TEMA_KEBUN),
    ("WEB HERO", TEMA_JARING_MERAH),
    ("NIGHT GUARDIAN", TEMA_MALAM_KOTA),
    ("SPEED HERO", TEMA_KILAT_BIRU),
    ("SHIELD HERO", TEMA_PERISAI_MERAH),
    ("WARRIOR HERO", TEMA_AMAZON),
    ("TANPA", None)
]


def gambar_overlay_tipe_khusus(canvas, pad, w, h, Wc, Hc, strip_bawah, tema):
    if tema is None:
        return
    tipe = tema.get("tipe")
    warna_teks = _warna_kontras(tema["bg"])

    if tipe == "musik":
        lagu = tema.get("lagu", "Song Title")
        artis = tema.get("artis", "Artist Name")
        lirik = tema.get("lirik", "")
        posisi = tema.get("posisi", "1:30")
        durasi = tema.get("durasi", "3:45")

        y_base = Hc - strip_bawah + 2
        box_x1 = pad + 4
        box_x2 = Wc - pad - 4
        box_y1 = y_base - 8
        box_y2 = y_base + 38

        # Background banner box besar & tebal untuk nama lagu
        cv2.rectangle(canvas, (box_x1, box_y1), (box_x2, box_y2), (18, 16, 24), cv2.FILLED)
        cv2.rectangle(canvas, (box_x1, box_y1), (box_x2, box_y2), tema["border"], 3, cv2.LINE_AA)

        # Judul lagu SANGAT BESAR & BOLD (scale 0.90)
        info_song = f"♫ {lagu.upper()}"
        info_artist = f"- {artis}"
        teks(canvas, info_song, (box_x1 + 12, box_y1 + 30), 0.90, 3, PUTIH)
        (tw, _), _ = cv2.getTextSize(info_song, FONT, 0.90, 3)
        teks(canvas, info_artist, (box_x1 + 24 + tw, box_y1 + 30), 0.65, 2, AMBER, aberasi=False)

        # Progress bar
        bar_y = box_y2 + 14
        bar_w = max(40, box_x2 - box_x1)
        cv2.line(canvas, (box_x1, bar_y), (box_x2, bar_y), (140, 140, 140), 2, cv2.LINE_AA)
        cv2.line(canvas, (box_x1, bar_y), (box_x1 + int(bar_w * 0.45), bar_y), tema["border"], 3, cv2.LINE_AA)
        cv2.circle(canvas, (box_x1 + int(bar_w * 0.45), bar_y), 5, PUTIH, cv2.FILLED, cv2.LINE_AA)

        time_text = f"{posisi} / {durasi}   |<<   >   >>|   ♥"
        cv2.putText(canvas, time_text, (box_x1, bar_y + 18), FONT, 0.42, warna_teks, 1, cv2.LINE_AA)

        if lirik:
            cv2.putText(canvas, lirik, (box_x1, Hc - 6), FONT, 0.45, tema["border"], 1, cv2.LINE_AA)

    elif tipe == "film_strip":
        hole_w, hole_h = 10, 14
        step = 26
        for y_hole in range(8, Hc - 8, step):
            cv2.rectangle(canvas, (pad // 2 - hole_w // 2, y_hole), (pad // 2 + hole_w // 2, y_hole + hole_h), (10, 10, 10), cv2.FILLED)
            cv2.rectangle(canvas, (pad // 2 - hole_w // 2, y_hole), (pad // 2 + hole_w // 2, y_hole + hole_h), (140, 140, 140), 1)
            cv2.rectangle(canvas, (Wc - pad // 2 - hole_w // 2, y_hole), (Wc - pad // 2 + hole_w // 2, y_hole + hole_h), (10, 10, 10), cv2.FILLED)
            cv2.rectangle(canvas, (Wc - pad // 2 - hole_w // 2, y_hole), (Wc - pad // 2 + hole_w // 2, y_hole + hole_h), (140, 140, 140), 1)
        cv2.putText(canvas, "35MM FILM STAGE  -  KODAK PORTRA 400", (pad, Hc - strip_bawah // 2 + 6), FONT, 0.45, warna_teks, 1, cv2.LINE_AA)

    elif tipe == "magazine":
        cv2.putText(canvas, "V O G U E  E D I T I O N", (pad + 6, pad - 5), FONT, 0.55, warna_teks, 1, cv2.LINE_AA)

    elif tipe == "cyber":
        cv2.putText(canvas, "[SYS::ONLINE] RADAR_SCAN=OK", (pad, pad - 5), FONT, 0.42, CYAN, 1, cv2.LINE_AA)

idx_bingkai = 0


def _warna_kontras(bg):
    lum = 0.114 * bg[0] + 0.587 * bg[1] + 0.299 * bg[2]
    return (30, 28, 32) if lum > 150 else PUTIH


def _dalam_kotak(x, y, kotak_list, margin=0):
    for (x1, y1, x2, y2) in kotak_list:
        if x1 - margin <= x <= x2 + margin and y1 - margin <= y <= y2 + margin:
            return True
    return False


def buat_lapisan_stiker(Wc, Hc, tema, kotak_aman, kepadatan=1.0):
    """Taburkan banyak stiker doodle di area BEBAS (di luar kotak_aman = area foto),
    kepadatan mengikuti luas area bebas supaya bingkai besar tetap ramai isinya."""
    rng = random.Random(int(time.time() * 977) + random.randint(0, 999))
    area_total = Wc * Hc
    area_terpakai = sum((x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in kotak_aman)
    area_bebas = max(400, area_total - area_terpakai)
    langkah = 30
    jumlah_target = min(70, max(10, int(area_bebas / (langkah * langkah) * kepadatan)))
    daftar = []
    coba, maks_coba = 0, jumlah_target * 25
    while len(daftar) < jumlah_target and coba < maks_coba:
        coba += 1
        x = rng.randint(6, Wc - 6)
        y = rng.randint(6, Hc - 6)
        if _dalam_kotak(x, y, kotak_aman, margin=8):
            continue
        if any(abs(x - d["pusat"][0]) < 18 and abs(y - d["pusat"][1]) < 18 for d in daftar):
            continue
        r = rng.randint(7, 15)
        fn = rng.choice(tema["stiker"])
        warna = rng.choice(tema["warna"])
        daftar.append({"fn": fn, "pusat": (x, y), "r": r, "warna": warna})
    return daftar


def gambar_stiker_dari_daftar(img, daftar, skala=1.0, alpha=1.0):
    for d in daftar:
        r = max(2, int(d["r"] * skala))
        if alpha >= 0.999:
            d["fn"](img, d["pusat"], r, d["warna"])
        else:
            ov = img.copy()
            d["fn"](ov, d["pusat"], r, d["warna"])
            cv2.addWeighted(ov, alpha, img, 1 - alpha, 0, img)


def _gambar_tape_sudut(canvas, kotak, tema):
    x1, y1, x2, y2 = kotak
    warna_list = tema.get("tape_warna", tema["warna"])

    def tape(cx, cy, warna, sudut):
        box = cv2.boxPoints(((cx, cy), (66, 20), sudut)).astype(np.int32)
        ov = canvas.copy()
        cv2.fillConvexPoly(ov, box, warna)
        cv2.addWeighted(ov, 0.6, canvas, 0.4, 0, canvas)
        cv2.polylines(canvas, [box], True, (255, 255, 255), 1, cv2.LINE_AA)

    tape(x1 + 6, y1 + 4, random.choice(warna_list), -18)
    tape(x2 - 6, y1 + 4, random.choice(warna_list), 18)
    tape(x1 + 6, y2 - 4, random.choice(warna_list), 18)
    tape(x2 - 6, y2 - 4, random.choice(warna_list), -18)


def bingkai_generic(img, tema):
    """Bangun satu foto jadi kartu berbingkai penuh hiasan. Mengembalikan
    (gambar_final, gambar_tanpa_stiker, daftar_stiker) - dua yang terakhir
    dipakai untuk versi GIF berkedip (opsional)."""
    if tema is None:
        return img, None, None
    h, w = img.shape[:2]
    pad = max(16, int(w * tema.get("pad", 0.11)))
    strip_bawah = max(78, int(h * 0.16)) if tema.get("tipe") == "musik" else max(46, int(h * 0.13))
    Wc, Hc = w + pad * 2, h + pad * 2 + strip_bawah
    canvas = np.full((Hc, Wc, 3), tema["bg"], np.uint8)
    canvas[pad:pad + h, pad:pad + w] = img
    cv2.rectangle(canvas, (2, 2), (Wc - 3, Hc - 3), tema["border"], 3, cv2.LINE_AA)
    cv2.rectangle(canvas, (pad - 2, pad - 2), (pad + w + 2, pad + h + 2), tema["border"], 2, cv2.LINE_AA)

    kotak_aman = [(pad, pad, pad + w, pad + h)]
    daftar_stiker = buat_lapisan_stiker(Wc, Hc, tema, kotak_aman, kepadatan=1.15)
    if tema.get("tape"):
        _gambar_tape_sudut(canvas, kotak_aman[0], tema)

    canvas_dasar = canvas.copy()
    gambar_stiker_dari_daftar(canvas, daftar_stiker)

    warna_teks = _warna_kontras(tema["bg"])
    judul = f'{tema.get("judul", "RETROLENS")}  {time.strftime("%d.%m.%Y")}'
    if tema.get("tipe") != "musik":
        cv2.putText(canvas, judul, (pad, Hc - strip_bawah // 2 + 6), FONT, 0.55,
                    warna_teks, 1, cv2.LINE_AA)
    gambar_overlay_tipe_khusus(canvas, pad, w, h, Wc, Hc, strip_bawah, tema)
    return canvas, canvas_dasar, daftar_stiker


# ====================== LIVE PREVIEW BINGKAI ======================
# Supaya bingkai kelihatan langsung di kamera (bukan cuma pas foto tersimpan),
# kita cache posisi stiker per tema (biar tidak "berkedip acak" tiap frame) dan
# hanya membangun ulang saat tema/ukuran berubah.

_bingkai_cache = {"idx": -1, "w": 0, "h": 0, "data": None}


def _bangun_cache_bingkai(tema, w, h):
    if tema is None:
        return None
    pad = max(16, int(w * tema.get("pad", 0.11)))
    strip_bawah = max(78, int(h * 0.16)) if tema.get("tipe") == "musik" else max(46, int(h * 0.13))
    Wc, Hc = w + pad * 2, h + pad * 2 + strip_bawah
    kotak_aman = [(pad, pad, pad + w, pad + h)]
    daftar_stiker = buat_lapisan_stiker(Wc, Hc, tema, kotak_aman, kepadatan=1.15)
    return dict(pad=pad, strip_bawah=strip_bawah, Wc=Wc, Hc=Hc,
                kotak_aman=kotak_aman, daftar_stiker=daftar_stiker)


def tampilkan_dengan_bingkai(img, idx):
    """Bungkus frame kamera langsung (live) dengan bingkai tema terpilih."""
    global _bingkai_cache
    nama_tema, tema = BINGKAI_LUCU[idx]
    if tema is None:
        return img
    h, w = img.shape[:2]
    if _bingkai_cache["idx"] != idx or _bingkai_cache["w"] != w or _bingkai_cache["h"] != h:
        _bingkai_cache = {"idx": idx, "w": w, "h": h,
                          "data": _bangun_cache_bingkai(tema, w, h)}
    data = _bingkai_cache["data"]
    if data is None:
        return img

    pad, strip_bawah, Wc, Hc = data["pad"], data["strip_bawah"], data["Wc"], data["Hc"]
    canvas = np.full((Hc, Wc, 3), tema["bg"], np.uint8)
    canvas[pad:pad + h, pad:pad + w] = img
    cv2.rectangle(canvas, (2, 2), (Wc - 3, Hc - 3), tema["border"], 3, cv2.LINE_AA)
    cv2.rectangle(canvas, (pad - 2, pad - 2), (pad + w + 2, pad + h + 2), tema["border"], 2, cv2.LINE_AA)
    if tema.get("tape"):
        _gambar_tape_sudut(canvas, data["kotak_aman"][0], tema)
    gambar_stiker_dari_daftar(canvas, data["daftar_stiker"])
    warna_teks = _warna_kontras(tema["bg"])
    judul = f'{tema.get("judul", "RETROLENS")}  {time.strftime("%d.%m.%Y")}'
    if tema.get("tipe") != "musik":
        cv2.putText(canvas, judul, (pad, Hc - strip_bawah // 2 + 6), FONT, 0.55,
                    warna_teks, 1, cv2.LINE_AA)
    gambar_overlay_tipe_khusus(canvas, pad, w, h, Wc, Hc, strip_bawah, tema)
    return canvas


# ====================== TATA LETAK GRID PHOTOBOOTH ======================
# Jumlah foto per strip sekarang bisa 2-6, dan tata letaknya menyesuaikan
# (bukan cuma ditumpuk vertikal ke bawah).

_POLA_GRID = {
    1: [1],
    2: [2],
    3: [2, 1],
    4: [2, 2],
    5: [2, 3],
    6: [3, 3],
}


def _pola_grid(n):
    return _POLA_GRID.get(n, [n])


def susun_grid(daftar_foto, tema, lebar_sel=210, gaya_layout="VERTIKAL"):
    """Susun beberapa foto dalam grid/strip (baris bisa berbeda jumlah kolomnya),
    dihias dengan tema bingkai yang sama seperti foto tunggal."""
    n = max(1, len(daftar_foto))
    if gaya_layout == "VERTIKAL":
        pola = [1] * n
    elif gaya_layout == "HORIZONTAL":
        pola = [n]
    else:  # GRID
        pola = _pola_grid(n)

    sel = []
    for f in daftar_foto:
        h0, w0 = f.shape[:2]
        skala = lebar_sel / w0
        sel.append(cv2.resize(f, (lebar_sel, max(1, int(h0 * skala)))))
    tinggi_sel = max(im.shape[0] for im in sel)
    sel = [im if im.shape[0] == tinggi_sel else
           cv2.copyMakeBorder(im, 0, tinggi_sel - im.shape[0], 0, 0,
                              cv2.BORDER_CONSTANT, value=(0, 0, 0)) for im in sel]

    gap, pad, strip_bawah = 14, 24, 56
    kolom_maks = max(pola)
    lebar_baris_maks = kolom_maks * lebar_sel + (kolom_maks - 1) * gap
    tinggi_total = len(pola) * tinggi_sel + (len(pola) - 1) * gap
    Wc = lebar_baris_maks + pad * 2
    Hc = tinggi_total + pad * 2 + strip_bawah

    bg = tema["bg"] if tema else (245, 245, 245)
    border = tema["border"] if tema else ABU
    canvas = np.full((Hc, Wc, 3), bg, np.uint8)

    kotak_aman = []
    idx_foto = 0
    y = pad
    for kolom_n in pola:
        lebar_baris_ini = kolom_n * lebar_sel + (kolom_n - 1) * gap
        x = pad + (lebar_baris_maks - lebar_baris_ini) // 2
        for _ in range(kolom_n):
            im = sel[idx_foto]
            idx_foto += 1
            canvas[y:y + tinggi_sel, x:x + lebar_sel] = im
            cv2.rectangle(canvas, (x - 1, y - 1), (x + lebar_sel + 1, y + tinggi_sel + 1),
                          border, 2, cv2.LINE_AA)
            kotak_aman.append((x, y, x + lebar_sel, y + tinggi_sel))
            x += lebar_sel + gap
        y += tinggi_sel + gap
    cv2.rectangle(canvas, (2, 2), (Wc - 3, Hc - 3), border, 3, cv2.LINE_AA)

    if tema is not None:
        daftar_stiker = buat_lapisan_stiker(Wc, Hc, tema, kotak_aman, kepadatan=1.3)
        if tema.get("tape"):
            for kotak in kotak_aman:
                _gambar_tape_sudut(canvas, kotak, tema)
        canvas_dasar = canvas.copy()
        gambar_stiker_dari_daftar(canvas, daftar_stiker)
        warna_teks = _warna_kontras(bg)
        judul = f'{tema.get("judul", "PHOTOBOOTH")} STRIP  {time.strftime("%d.%m.%Y %H:%M")}'
        cv2.putText(canvas, judul, (pad, Hc - strip_bawah // 2 + 6), FONT, 0.5,
                    warna_teks, 1, cv2.LINE_AA)
        gambar_overlay_tipe_khusus(canvas, pad, 210, 150, Wc, Hc, strip_bawah, tema)
    else:
        canvas_dasar, daftar_stiker = canvas.copy(), []

    return canvas, canvas_dasar, daftar_stiker


def buat_gif_kilau(dasar, daftar_stiker, path_png, n_frame=8, durasi_ms=90):
    """Bonus: simpan versi .gif animasi di mana stiker berkedip/berdenyut
    pelan (twinkle). Best-effort - kalau Pillow tidak ada, cukup dilewati."""
    if not daftar_stiker:
        return
    try:
        from PIL import Image
    except ImportError:
        print("(Pillow belum terpasang - lewati GIF animasi. `pip install pillow` untuk mengaktifkan.)")
        return
    frame_pil = []
    for i in range(n_frame):
        fase = (i / n_frame) * 2 * math.pi
        kanvas = dasar.copy()
        for j, d in enumerate(daftar_stiker):
            offset = j * 0.7
            kedip = 0.55 + 0.45 * math.sin(fase + offset)
            skala = 0.7 + 0.5 * max(0.0, kedip)
            alpha = max(0.25, kedip)
            r = max(2, int(d["r"] * skala))
            ov = kanvas.copy()
            d["fn"](ov, d["pusat"], r, d["warna"])
            cv2.addWeighted(ov, alpha, kanvas, 1 - alpha, 0, kanvas)
        rgb = cv2.cvtColor(kanvas, cv2.COLOR_BGR2RGB)
        frame_pil.append(Image.fromarray(rgb))
    path_gif = os.path.splitext(path_png)[0] + ".gif"
    try:
        frame_pil[0].save(path_gif, save_all=True, append_images=frame_pil[1:],
                          duration=durasi_ms, loop=0)
        print("GIF animasi tersimpan:", path_gif)
    except Exception as e:
        print("Gagal membuat GIF animasi:", e)


# --- SISTEM MUSIK (pygame.mixer) ---
os.makedirs(MUSIK_FOLDER, exist_ok=True)
pygame.mixer.init()
_musik_sedang_putar = False


def musik_putar(folder=MUSIK_FOLDER):
    """Cari & putar file musik pertama di folder (loop). Mendukung mp3/wav/ogg."""
    global _musik_sedang_putar
    if _musik_sedang_putar:
        return
    ekstensi = (".mp3", ".wav", ".ogg")
    for f in sorted(os.listdir(folder)):
        if f.lower().endswith(ekstensi):
            path = os.path.join(folder, f)
            try:
                pygame.mixer.music.load(path)
                pygame.mixer.music.set_volume(MUSIK_VOLUME)
                pygame.mixer.music.play(-1)  # -1 = loop tak terbatas
                _musik_sedang_putar = True
                print(f"Musik diputar: {path}")
            except Exception as e:
                print(f"Gagal putar musik: {e}")
            return
    print(f"Tidak ada file musik di folder '{folder}/'")


def musik_hentikan():
    """Hentikan musik yang sedang diputar."""
    global _musik_sedang_putar
    try:
        pygame.mixer.music.stop()
    except:
        pass
    _musik_sedang_putar = False


def label_blur_mode(tampil, blur_aktif, blur_kuat):
    """Label statis (tanpa animasi) untuk menandai status mode BLUR."""
    h, w = tampil.shape[:2]
    label = "BLUR MODE" if not blur_aktif else "BLUR AKTIF"
    (tw, th), _ = cv2.getTextSize(label, FONT, 0.9, 2)
    teks(tampil, label, (w // 2 - tw // 2, 60), 0.9, 2, MAGENTA)

    if blur_aktif:
        level_pct = min(100, int((blur_kuat / BLUR_KUAT_MAX) * 100))
        bar_total_w = 180
        bar_filled = int(bar_total_w * level_pct / 100)
        bx, by = w - bar_total_w - 30, h - 80
        cv2.rectangle(tampil, (bx, by), (bx + bar_total_w, by + 14), SAMAR, cv2.FILLED)
        cv2.rectangle(tampil, (bx, by), (bx + bar_filled, by + 14), MAGENTA, cv2.FILLED)
        cv2.rectangle(tampil, (bx, by), (bx + bar_total_w, by + 14), PUTIH, 1)
        teks(tampil, f"BLUR {level_pct}%", (bx, by - 8), 0.45, 1, MAGENTA, aberasi=False)

    return tampil


def progres_tahan(aktif_sekarang, progress, absen, dt, toleransi=TOLERANSI_GESTUR):
    """Progress gestur-tahan dengan toleransi: kehilangan deteksi sesaat tidak
    langsung mereset progress ke 0, cuma dijeda sampai batas toleransi frame."""
    if aktif_sekarang:
        absen = 0
        progress += dt
    else:
        absen += 1
        if absen > toleransi:
            progress = 0.0
    return progress, absen


class KameraStream:
    def __init__(self, sumber):
        if isinstance(sumber, int):
            self.cap = cv2.VideoCapture(sumber, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(sumber)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.frame = None
        self.jalan = True
        self.lock = threading.Lock()
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.jalan:
            ok, f = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self.lock:
                self.frame = f

    def baca(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self):
        self.jalan = False
        time.sleep(0.15)
        self.cap.release()

opsi = vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=MODEL),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.35,   # diturunkan agar gestur parsial (peace, kepal)
    min_hand_presence_confidence=0.35,    # lebih gampang lolos deteksi ulang, tidak
    min_tracking_confidence=0.35,         # harus nunjukin telapak terbuka dulu
)
landmarker = vision.HandLandmarker.create_from_options(opsi)

kamera = KameraStream(SUMBER)
time.sleep(1.0)
if not kamera.cap.isOpened():
    print("Kamera gagal dibuka. Cek IP / kabel / WiFi.")
    raise SystemExit

cv2.namedWindow("RETROLENS", cv2.WINDOW_NORMAL)
cv2.resizeWindow("RETROLENS", LEBAR, TINGGI)
KONEKSI = vision.HandLandmarksConnections.HAND_CONNECTIONS

mode = "LENSA"
idx_lensa = 0
quad = None
hilang = 99
mulai_diam = None
mulai_ganti = None
kilat_sampai = 0.0
jml_foto = 0
prev_time = 0.0
waktu_lalu = 0.0
waktu_mulai = time.time()
ts = 0
debug = False

zoom_level = 1.0
zoom_target = 1.0
zoom_aktif = False
zoom_streak = 0
zoom_pusat = np.array([LEBAR / 2, TINGGI / 2], dtype=np.float32)

idx_efek_peace = 0
efek_arming_progress = 0.0    # progress fase konfirmasi (tahan 3 detik)
efek_arming_absen = 0
efek_terpicu = False          # sudah lolos konfirmasi -> hitung mundur foto jalan sendiri
efek_mulai = None
efek_posisi = None

kanvas = np.zeros((TINGGI, LEBAR, 3), np.uint8)
pena_akhir = None
pena_halus = None
menggambar = False
beruntun_ya = beruntun_tidak = 0
goresan = 0
PENA = [PUTIH, CYAN, MAGENTA, AMBER]
idx_pena = 1
strokes = []            # daftar coretan, tiap elemen: {"warna":.., "titik":[..]}
tumpukan_kubus = []     # tumpukan kubus 3D di udara

thumbnail_terakhir = None
thumbnail_waktu = 0.0
bantuan = False

# --- STATE BLUR MODE ---
blur_mode_kuat = 1.0              # kekuatan blur saat ini (1 = normal)
blur_mode_target = 1.0            # target blur
tiga_jari_progress = 0.0          # progress masuk mode blur
tiga_jari_absen = 0
blur_peace_absen = 0              # toleransi kedip deteksi ✌ di dalam mode BLUR
blur_peace_terlihat = False       # status ✌ yang sudah "distabilkan" (anti-kedip)

# --- STATE PHOTOBOOTH (strip beberapa foto sekaligus) ---
JUMLAH_BOOTH = 4                  # jumlah foto per strip (bisa diubah tombol [ ])
BOOTH_MIN, BOOTH_MAKS = 2, 6
TAHAN_BOOTH = 2.0                 # detik hitung mundur tiap jepretan booth
booth_aktif = False
booth_daftar = []
booth_hitung_mulai = None

BOOTH_LAYOUTS = ["VERTIKAL", "GRID", "HORIZONTAL"]
idx_booth_layout = 0
menu_bingkai_terbuka = False


def _simpan_file(gambar, tag):
    global jml_foto, kilat_sampai, thumbnail_terakhir, thumbnail_waktu
    nama = os.path.join(FOLDER, time.strftime(f"{tag}_%Y%m%d_%H%M%S.png"))
    cv2.imwrite(nama, gambar)
    print("Tersimpan:", nama, f"{gambar.shape[1]}x{gambar.shape[0]}")
    jml_foto += 1
    kilat_sampai = time.time() + 0.30
    thumbnail_terakhir = cv2.resize(gambar, (THUMB_W, THUMB_H))
    thumbnail_waktu = time.time()
    return nama


def simpan(gambar, tag="LENS", pakai_bingkai=True):
    dasar = daftar_stiker = None
    if pakai_bingkai:
        _, tema = BINGKAI_LUCU[idx_bingkai]
        if tema is not None:
            gambar, dasar, daftar_stiker = bingkai_generic(gambar, tema)
    nama = _simpan_file(gambar, tag)
    if dasar is not None:
        buat_gif_kilau(dasar, daftar_stiker, nama)


def simpan_strip_booth(daftar_foto, tag="BOOTH"):
    _, tema = BINGKAI_LUCU[idx_bingkai]
    gaya = BOOTH_LAYOUTS[idx_booth_layout]
    canvas, dasar, daftar_stiker = susun_grid(daftar_foto, tema, gaya_layout=gaya)
    nama = _simpan_file(canvas, tag)
    if tema is not None:
        buat_gif_kilau(dasar, daftar_stiker, nama)


def gambar_menu_pemilih_bingkai(tampil, idx_aktif, terbuka):
    if not terbuka:
        return tampil
    h, w = tampil.shape[:2]
    ov = tampil.copy()
    box_x1, box_y1, box_x2, box_y2 = 40, 30, w - 40, h - 30
    cv2.rectangle(ov, (box_x1, box_y1), (box_x2, box_y2), (18, 16, 24), cv2.FILLED)
    tampil = cv2.addWeighted(ov, 0.90, tampil, 0.10, 0)
    cv2.rectangle(tampil, (box_x1, box_y1), (box_x2, box_y2), MAGENTA, 2, cv2.LINE_AA)
    cv2.rectangle(tampil, (box_x1 + 3, box_y1 + 3), (box_x2 - 3, box_y2 - 3), CYAN, 1, cv2.LINE_AA)

    teks(tampil, "MENU PEMILIH BINGKAI & TEMA PHOTOBOOTH", (box_x1 + 20, box_y1 + 32), 0.65, 2, CYAN)
    teks(tampil, "Navigasi: [< / >] atau [A / D] pilih bingkai | [F] / [ENTER] Konfirmasi / Tutup",
         (box_x1 + 20, box_y1 + 54), 0.44, 1, ABU, aberasi=False)

    nama_tema_aktif, tema_aktif = BINGKAI_LUCU[idx_aktif]
    kategori_map = {"LAGU": "🎵 TEMA LAGU", "VINTAGE": "📼 RETRO / VINTAGE", "LUCU": "✨ ESTETIKA LUCU", "HERO": "🦸 SUPERHERO"}

    y_pos = box_y1 + 75
    col_w = (box_x2 - box_x1 - 40) // 2

    # Left Column: Frame List
    cv2.rectangle(tampil, (box_x1 + 20, y_pos), (box_x1 + 20 + col_w, box_y2 - 20), (28, 25, 36), cv2.FILLED)
    cv2.rectangle(tampil, (box_x1 + 20, y_pos), (box_x1 + 20 + col_w, box_y2 - 20), SAMAR, 1)
    teks(tampil, "DAFTAR TEMA BINGKAI", (box_x1 + 30, y_pos + 24), 0.52, 2, PUTIH, aberasi=False)

    start_i = max(0, min(idx_aktif - 4, len(BINGKAI_LUCU) - 9))
    start_i = max(0, start_i)

    for i in range(start_i, min(len(BINGKAI_LUCU), start_i + 9)):
        nama_t, t_obj = BINGKAI_LUCU[i]
        curr_y = y_pos + 52 + (i - start_i) * 30
        terpilih = (i == idx_aktif)

        if terpilih:
            cv2.rectangle(tampil, (box_x1 + 25, curr_y - 18), (box_x1 + 15 + col_w, curr_y + 8), MAGENTA, cv2.FILLED)
            teks(tampil, f"> {i+1}. {nama_t}", (box_x1 + 32, curr_y), 0.48, 2, PUTIH, aberasi=False)
        else:
            teks(tampil, f"  {i+1}. {nama_t}", (box_x1 + 32, curr_y), 0.45, 1, ABU, aberasi=False)

    # Right Column: Preview Detail Card
    right_x = box_x1 + 40 + col_w
    cv2.rectangle(tampil, (right_x, y_pos), (box_x2 - 20, box_y2 - 20), (28, 25, 36), cv2.FILLED)
    cv2.rectangle(tampil, (right_x, y_pos), (box_x2 - 20, box_y2 - 20), AMBER, 1)

    teks(tampil, "DETAIL TEMA TERPILIH", (right_x + 15, y_pos + 24), 0.52, 2, AMBER, aberasi=False)

    if tema_aktif is not None:
        teks(tampil, f"Judul: {tema_aktif.get('judul', nama_tema_aktif)}", (right_x + 15, y_pos + 60), 0.48, 2, PUTIH, aberasi=False)
        kateg = tema_aktif.get("kateg", "LUCU")
        teks(tampil, f"Kategori: {kategori_map.get(kateg, kateg)}", (right_x + 15, y_pos + 90), 0.45, 1, CYAN, aberasi=False)

        if tema_aktif.get("tipe") == "musik":
            teks(tampil, f"Lagu: {tema_aktif.get('lagu')}", (right_x + 15, y_pos + 120), 0.45, 1, MAGENTA, aberasi=False)
            teks(tampil, f"Artis: {tema_aktif.get('artis')}", (right_x + 15, y_pos + 145), 0.45, 1, PUTIH, aberasi=False)
            teks(tampil, f"Lirik: {tema_aktif.get('lirik')}", (right_x + 15, y_pos + 175), 0.40, 1, AMBER, aberasi=False)
        else:
            teks(tampil, f"Fitur: Tape={tema_aktif.get('tape', False)}, Stiker={len(tema_aktif.get('stiker', []))}",
                 (right_x + 15, y_pos + 120), 0.45, 1, PUTIH, aberasi=False)

        bg_col = tema_aktif.get("bg", (0, 0, 0))
        border_col = tema_aktif.get("border", (255, 255, 255))
        cv2.rectangle(tampil, (right_x + 15, y_pos + 215), (right_x + 85, y_pos + 255), bg_col, cv2.FILLED)
        cv2.rectangle(tampil, (right_x + 15, y_pos + 215), (right_x + 85, y_pos + 255), border_col, 3)
        teks(tampil, "Warna Latar & Border", (right_x + 95, y_pos + 240), 0.44, 1, ABU, aberasi=False)
    else:
        teks(tampil, "Tanpa Bingkai (Kamera Polos)", (right_x + 15, y_pos + 80), 0.50, 1, ABU, aberasi=False)

    return tampil


def gambar_ulang_kanvas():
    """Bangun ulang kanvas dari daftar strokes (dipakai untuk undo)."""
    kanvas[:] = 0
    for s in strokes:
        pts = s["titik"]
        for i in range(1, len(pts)):
            cv2.line(kanvas, pts[i - 1], pts[i], s["warna"], 6, cv2.LINE_AA)

while True:
    frame = kamera.baca()
    if frame is None:
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (LEBAR, TINGGI))
    bersih = frame                       
    w, h = LEBAR, TINGGI
    now = time.time()
    dt = min(now - waktu_lalu, 0.1) if waktu_lalu else 0.0
    waktu_lalu = now

    rgb = cv2.cvtColor(bersih, cv2.COLOR_BGR2RGB)
    # Timestamp video pakai waktu ASLI (ms sejak program mulai), bukan asumsi
    # 30fps tetap - supaya tracking MediaPipe akurat walau frame kadang berat
    # (mis. saat blur gaussian di mode BLUR bikin frame time > 33ms).
    ts_baru = int((now - waktu_mulai) * 1000)
    ts = ts_baru if ts_baru > ts else ts + 1  # tetap harus naik terus (monoton)
    hasil = landmarker.detect_for_video(
        mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)
    tangan = hasil.hand_landmarks

    info = []
    for hand in tangan:
        f = jari_terbuka(hand)
        px = [(int(l.x * w), int(l.y * h)) for l in hand]
        info.append((hand, f, px))

    nama_lensa, fn_lensa = LENSA[idx_lensa]
    status = "-"

    if mode == "LENSA":
        if booth_aktif:
            tampil = grade_retro(bersih)
            zoom_aktif = False
            zoom_streak = 0
            efek_terpicu = False
            efek_arming_progress = efek_arming_absen = 0
            if booth_hitung_mulai is None:
                booth_hitung_mulai = now
            sisa_booth = TAHAN_BOOTH - (now - booth_hitung_mulai)
            pusat_booth = (w // 2, h // 2)
            maju_booth = min(1.0, max(0.0, 1 - sisa_booth / TAHAN_BOOTH))
            cincin(tampil, pusat_booth, 74, maju_booth, MAGENTA, 6)
            teks(tampil, f"BOOTH {len(booth_daftar) + 1}/{JUMLAH_BOOTH}",
                 (pusat_booth[0] - 118, pusat_booth[1] - 98), 0.7, 2, MAGENTA)
            teks(tampil, "bersiap ya... (ESC batal)",
                 (pusat_booth[0] - 138, pusat_booth[1] + 112), 0.55, 2, ABU)
            if sisa_booth <= 0:
                booth_daftar.append(fn_lensa(bersih, now))
                booth_hitung_mulai = now
                kilat_sampai = now + 0.15
                if len(booth_daftar) >= JUMLAH_BOOTH:
                    simpan_strip_booth(booth_daftar)
                    booth_aktif = False
                    booth_daftar = []
                    booth_hitung_mulai = None
            status = "BOOTH"
        else:
            hands_L = [(hand, f, px) for hand, f, px in info if is_L(f)]

            # --- ZOOM: aktif kalau HANYA SATU tangan bentuk L, stabil beberapa frame,
            #     DAN tidak sedang ada sesi bingkai 2-tangan yang baru berjalan,
            #     DAN tidak ada tangan yang sedang membentuk peace (✌).
            #     (quad/hilang di sini masih nilai dari frame SEBELUMNYA)
            bingkai_sedang_jalan = quad is not None and hilang < 6
            ada_peace = any(is_peace(f) for _, f, _ in info)

            if len(hands_L) == 1 and not bingkai_sedang_jalan and not ada_peace:
                zoom_streak += 1
                _, _, px_zoom = hands_L[0]
                # titik tengah antara ujung jempol & telunjuk = "titik cubitan"
                target_pusat = np.array(
                    [(px_zoom[4][0] + px_zoom[8][0]) / 2,
                     (px_zoom[4][1] + px_zoom[8][1]) / 2], dtype=np.float32)
                zoom_pusat += ZOOM_PUSAT_HALUS * (target_pusat - zoom_pusat)
            else:
                zoom_streak = 0

            if zoom_streak >= ZOOM_STABIL_FRAME:
                zoom_aktif = True
                rasio = rasio_pinch(hands_L[0][0], w, h)
                rasio = max(RASIO_PINCH_MIN, min(RASIO_PINCH_MAX, rasio))
                maju_zoom = (rasio - RASIO_PINCH_MIN) / (RASIO_PINCH_MAX - RASIO_PINCH_MIN)
                zoom_target = ZOOM_MIN + maju_zoom * (ZOOM_MAX - ZOOM_MIN)
            else:
                zoom_aktif = False
            zoom_level += ZOOM_HALUS * (zoom_target - zoom_level)

            # --- BINGKAI LENSA: aktif kalau DUA tangan bentuk L (seperti semula) ---
            sudut = []
            if len(hands_L) >= 2:
                for hand, f, px in hands_L:
                    sudut.append(px[4])
                    sudut.append(px[8])

            mentah = cocokkan(urutkan_quad(sudut[:4]), quad) if len(sudut) >= 4 else None

            geser_maks = 0.0
            if mentah is not None:
                if quad is None:
                    quad = mentah
                else:
                    geser_maks = float(np.max(np.linalg.norm(mentah - quad, axis=1)))
                    quad = quad + HALUS * (mentah - quad)
                hilang = 0
            else:
                hilang += 1

            tampil = grade_retro(bersih)

            aktif = hilang < 6 and quad is not None
            if aktif:
                q = quad.astype(np.float32)
                diag = (np.linalg.norm(q[2] - q[0]) + np.linalg.norm(q[3] - q[1])) / 2

                if diag < MIN_BUKA:
                    status = "GENGGAM"
                    mulai_diam = None
                    c = titik_int(q.mean(axis=0))
                    r = int(18 + 6 * math.sin(now * 6))
                    cv2.circle(tampil, c, r, CYAN, 2, cv2.LINE_AA)
                    cv2.circle(tampil, c, 3, CYAN, cv2.FILLED)
                    teks(tampil, "TARIK UNTUK MEMBUKA", (c[0] - 128, c[1] - 40), 0.6, 2)
                else:

                    efek, balik, meta = warp_efek(bersih, q, fn_lensa, now)
                    if efek is not None:
                        tampil = komposit(tampil, balik, meta)

                        if geser_maks > GOYANG or mulai_diam is None:
                            mulai_diam = now
                        sisa = TAHAN_FOTO - (now - mulai_diam)
                        status = "ATUR" if geser_maks > GOYANG else "DIAM"
                        warna = AMBER if sisa > 1 else MAGENTA

                        u = (now * 0.35) % 1.0
                        cv2.line(tampil, titik_int(q[0] + (q[3] - q[0]) * u),
                                 titik_int(q[1] + (q[2] - q[1]) * u), CYAN, 1, cv2.LINE_AA)
                        cv2.polylines(tampil, [q.astype(np.int32)], True, SAMAR, 1,
                                      cv2.LINE_AA)
                        kurung_quad(tampil, q, warna)

                        n = q[0] + (q[1] - q[0]) * 0.02
                        teks(tampil, f"{nama_lensa}  {efek.shape[1]}x{efek.shape[0]}",
                             (int(n[0]), max(18, int(n[1]) - 12)), 0.5, 1, warna)

                        maju = min(1.0, max(0.0, 1 - sisa / TAHAN_FOTO))
                        a, b = q[3], q[2]
                        cv2.line(tampil, titik_int(a), titik_int(b), SAMAR, 4, cv2.LINE_AA)
                        cv2.line(tampil, titik_int(a), titik_int(a + (b - a) * maju),
                                 warna, 4, cv2.LINE_AA)

                        if sisa <= 0:
                            simpan(efek, nama_lensa)
                            mulai_diam = None
                            hilang = 99
                            quad = None
                        elif sisa < TAHAN_FOTO - 0.25:
                            ang = str(int(math.ceil(sisa)))
                            sk = 2.4 + 0.4 * abs(math.sin(sisa * math.pi))
                            (tw, th), _ = cv2.getTextSize(ang, FONT, sk, 6)
                            c = q.mean(axis=0)
                            teks(tampil, ang, (int(c[0] - tw / 2), int(c[1] + th / 2)),
                                 sk, 6, warna)
            else:
                mulai_diam = None
                if hilang > 20:
                    quad = None

            pemicu = None
            pemicu_blur = None
            for hand, f, px in info:
                if is_tunjuk_ketat(f):
                    pemicu = px[8]
                elif is_tiga_jari(f):
                    pemicu_blur = px[12]  # ujung jari tengah sebagai pusat cincin

            # --- MASUK MODE GAMBAR: telunjuk ketat ditahan ---
            if pemicu and not aktif and not pemicu_blur:
                if mulai_ganti is None:
                    mulai_ganti = now
                maju = (now - mulai_ganti) / TAHAN_MODE
                cincin(tampil, pemicu, 26, maju, CYAN)
                teks(tampil, "MODE GAMBAR", (pemicu[0] - 68, pemicu[1] - 42), 0.55, 2)
                if maju >= 1.0:
                    mode, mulai_ganti, quad = "GAMBAR", None, None
                    pena_akhir = pena_halus = None
                    menggambar = False
                    beruntun_ya = beruntun_tidak = 0
            # --- MASUK MODE BLUR: 3 jari (telunjuk+tengah+manis) ditahan ---
            elif pemicu_blur and not aktif:
                tiga_jari_progress, tiga_jari_absen = progres_tahan(
                    True, tiga_jari_progress, tiga_jari_absen, dt)
                maju_blur3 = tiga_jari_progress / TAHAN_MODE
                cincin(tampil, pemicu_blur, 30, maju_blur3, (255, 180, 50))  # warna emas
                teks(tampil, "MODE BLUR", (pemicu_blur[0] - 58, pemicu_blur[1] - 46), 0.55, 2)
                if maju_blur3 >= 1.0:
                    mode = "BLUR"
                    mulai_ganti = None
                    quad = None
                    tiga_jari_progress = tiga_jari_absen = 0
                    blur_mode_kuat = 1.0
                    blur_mode_target = 1.0
                    blur_peace_absen = 0
                    blur_peace_terlihat = False
                    musik_putar()
            else:
                mulai_ganti = None
                tiga_jari_progress, tiga_jari_absen = progres_tahan(
                    False, tiga_jari_progress, tiga_jari_absen, dt)

            # --- FOTO EFEK: gestur peace (✌) harus DITAHAN TAHAN_MULAI_EFEK detik
            #     dulu sebagai konfirmasi (anti kepicu tak sengaja). Setelah lolos,
            #     baru hitung mundur foto (TAHAN_BLUR) berjalan SENDIRI - tangan
            #     boleh diturunkan - supaya hasil foto tidak ikut kepotret gesturnya.
            nama_efek, fn_efek = EFEK_PEACE[idx_efek_peace]
            peace_px = None
            for hand, f, px in info:
                if is_peace(f):
                    peace_px = (int((px[8][0] + px[12][0]) / 2),
                                int((px[8][1] + px[12][1]) / 2))
                    break

            if efek_terpicu:
                # --- Fase 2: hitung mundur foto, tidak butuh tangan lagi ---
                sisa_efek = TAHAN_BLUR - (now - efek_mulai)
                maju_blur = min(1.0, max(0.0, 1 - sisa_efek / TAHAN_BLUR))
                cincin(tampil, efek_posisi, 30, maju_blur, MAGENTA)
                teks(tampil, f"FOTO {nama_efek}",
                     (efek_posisi[0] - 58, efek_posisi[1] - 46), 0.55, 2)
                if sisa_efek <= 0:
                    simpan(fn_efek(bersih), nama_efek)
                    efek_terpicu = False
            elif not aktif:
                # --- Fase 1: gestur ✌ harus ditahan dulu sebagai konfirmasi ---
                efek_arming_progress, efek_arming_absen = progres_tahan(
                    bool(peace_px), efek_arming_progress, efek_arming_absen, dt)
                if efek_arming_progress > 0:
                    titik_arming = peace_px if peace_px else titik_int(zoom_pusat)
                    maju_arming = efek_arming_progress / TAHAN_MULAI_EFEK
                    cincin(tampil, titik_arming, 30, maju_arming, AMBER)
                    teks(tampil, f"TAHAN untuk {nama_efek}",
                         (titik_arming[0] - 90, titik_arming[1] - 46), 0.5, 2, AMBER)
                    if maju_arming >= 1.0:
                        efek_terpicu = True
                        efek_mulai = now
                        efek_posisi = titik_arming
                        efek_arming_progress = efek_arming_absen = 0
            else:
                efek_arming_progress = efek_arming_absen = 0

            for hand, f, px in info:
                ok = is_L(f)
                for i in (4, 8):
                    cv2.circle(tampil, px[i], 7, CYAN if ok else ABU,
                               2 if ok else 1, cv2.LINE_AA)

            if not tangan:
                teks(tampil, f"L=ZOOM  LL=BINGKAI  3J=BLUR  ✌={nama_efek}(E)  "
                              f"F=GANTI BINGKAI  T=PHOTOBOOTH  (H=BANTUAN)",
                     (22, h - 58), 0.5, 2, ABU)

    elif mode == "GAMBAR":
        tampil = cv2.multiply(grade_retro(bersih), 0.5, dtype=cv2.CV_8U)
        status = "SIAP"
        zoom_aktif = False
        zoom_streak = 0
        efek_terpicu = False
        efek_arming_progress = efek_arming_absen = 0

        keluar = None
        for hand, f, px in info:
            if is_telapak(f):
                keluar = px[9]
                break

        if keluar:
            if mulai_ganti is None:
                mulai_ganti = now
            maju = (now - mulai_ganti) / TAHAN_MODE
            cincin(tampil, keluar, 30, maju, AMBER)
            teks(tampil, "MODE LENSA", (keluar[0] - 62, keluar[1] - 46), 0.55, 2)
            status = "KELUAR"
            menggambar = False
            pena_akhir = pena_halus = None
            if maju >= 1.0:
                mode, mulai_ganti, quad, hilang = "LENSA", None, None, 99
        else:
            mulai_ganti = None

            hapus = False
            gambar_ok = False
            ujung = None
            if info:
                hand, f, px = info[0]
                ujung = px[8]
                if is_kepal(f):
                    hapus = True
                elif is_peace(f):
                    status = "PINDAH"
                elif is_tunjuk(f):
                    gambar_ok = True

            if hapus:
                kanvas[:] = 0
                strokes.clear()
                goresan = 0
                menggambar = False
                pena_akhir = pena_halus = None
                beruntun_ya = beruntun_tidak = 0
                status = "HAPUS"
            else:
                if gambar_ok:
                    beruntun_ya += 1
                    beruntun_tidak = 0
                else:
                    beruntun_tidak += 1
                    beruntun_ya = 0

                if not menggambar and beruntun_ya >= MULAI_BUTUH:
                    menggambar = True
                elif menggambar and beruntun_tidak >= HENTI_BUTUH:
                    menggambar = False
                    pena_akhir = pena_halus = None

                if menggambar and gambar_ok and ujung is not None:
                    p = np.array(ujung, dtype=np.float32)
                    pena_halus = p if pena_halus is None else \
                        pena_halus + HALUS_PENA * (p - pena_halus)
                    titik = titik_int(pena_halus)
                    if pena_akhir is None:
                        strokes.append({"warna": PENA[idx_pena], "titik": [titik]})
                    if pena_akhir is not None:
                        d = abs(titik[0] - pena_akhir[0]) + abs(titik[1] - pena_akhir[1])
                        if d < LOMPAT_MAKS:
                            cv2.line(kanvas, pena_akhir, titik, PENA[idx_pena],
                                     6, cv2.LINE_AA)
                            strokes[-1]["titik"].append(titik)
                            goresan += 1
                    pena_akhir = titik
                    status = "GAMBAR"
                    cv2.circle(tampil, titik, 12, PENA[idx_pena], 2, cv2.LINE_AA)
                elif ujung is not None:
                    cv2.circle(tampil, ujung, 9, ABU, 1, cv2.LINE_AA)

        for hand, f, px in info:
            for i in (4, 8, 12, 16, 20):
                cv2.circle(tampil, px[i], 4, (60, 60, 235), cv2.FILLED)

        kecil = cv2.GaussianBlur(cv2.resize(kanvas, (w // 3, h // 3)), (0, 0), 4)
        tampil = cv2.add(tampil, cv2.resize(kecil, (w, h)))
        tampil = cv2.add(tampil, kanvas)
        teks(tampil, f"{status}   GORESAN {goresan}", (22, h - 58), 0.6, 2, ABU)

    # ====================== MODE BLUR ======================
    elif mode == "BLUR":
        status = "BLUR"
        zoom_aktif = False
        zoom_streak = 0
        efek_terpicu = False
        efek_arming_progress = efek_arming_absen = 0

        # --- Deteksi gestur peace untuk mengaktifkan blur ---
        blur_peace_aktif = False
        for hand, f, px in info:
            if is_peace(f):
                blur_peace_aktif = True
                break

        # --- Stabilkan deteksi: toleransi beberapa frame kalau sesaat
        #     tidak kedeteksi (kedipan tracking), supaya blur tidak
        #     langsung mati-nyala tiap kali 1 frame gagal deteksi ---
        if blur_peace_aktif:
            blur_peace_absen = 0
            blur_peace_terlihat = True
        else:
            blur_peace_absen += 1
            if blur_peace_absen > TOLERANSI_GESTUR:
                blur_peace_terlihat = False

        # --- Atur target blur berdasarkan peace (versi stabil) ---
        if blur_peace_terlihat:
            blur_mode_target = float(BLUR_KUAT_MAX)
            status = "BLUR AKTIF"
        else:
            blur_mode_target = float(BLUR_KUAT_MIN)

        # --- Transisi halus blur ---
        blur_mode_kuat += BLUR_TRANSISI * (blur_mode_target - blur_mode_kuat)
        blur_mode_kuat = max(1.0, blur_mode_kuat)

        # --- Terapkan grade retro + blur ---
        tampil = grade_retro(bersih)
        if blur_mode_kuat > 2.0:
            k = int(blur_mode_kuat) | 1  # pastikan ganjil
            tampil = cv2.GaussianBlur(tampil, (k, k), 0)

        # --- Label status (statis, tanpa animasi) ---
        tampil = label_blur_mode(tampil, blur_mode_kuat > 2.0, blur_mode_kuat)

        # --- Gambar landmark tangan ---
        for hand, f, px in info:
            for i in (4, 8, 12, 16, 20):
                cv2.circle(tampil, px[i], 5,
                           MAGENTA if is_peace(f) else ABU,
                           cv2.FILLED if is_peace(f) else 1, cv2.LINE_AA)

        # --- Keluar mode BLUR: telapak terbuka ditahan ---
        keluar_blur = None
        for hand, f, px in info:
            if is_telapak(f):
                keluar_blur = px[9]
                break

        if keluar_blur:
            if mulai_ganti is None:
                mulai_ganti = now
            maju = (now - mulai_ganti) / TAHAN_MODE
            cincin(tampil, keluar_blur, 30, maju, AMBER)
            teks(tampil, "KELUAR BLUR", (keluar_blur[0] - 68, keluar_blur[1] - 46), 0.55, 2)
            status = "KELUAR"
            if maju >= 1.0:
                musik_hentikan()
                mode = "LENSA"
                mulai_ganti = None
                quad = None
                hilang = 99
                blur_mode_kuat = 1.0
                blur_mode_target = 1.0
                blur_peace_absen = 0
                blur_peace_terlihat = False
        else:
            mulai_ganti = None

        if not tangan:
            teks(tampil, "✌=BLUR ON  ✋=KELUAR  (B=TOGGLE)",
                 (22, h - 58), 0.5, 2, ABU)

    teks(tampil, time.strftime("%d.%m.%Y %H:%M:%S"), (w - 320, 38), 0.6, 2)

    fps = 1 / (now - prev_time) if prev_time else 0
    prev_time = now
    if mode == "BLUR":
        teks(tampil, f"BLUR MODE | FPS {int(fps)} | FOTO {jml_foto}",
             (22, h - 24), 0.55, 2)
    else:
        teks(tampil, f"{mode} | {nama_lensa} | EFEK {EFEK_PEACE[idx_efek_peace][0]} | "
                      f"BINGKAI {BINGKAI_LUCU[idx_bingkai][0]} (F) | BOOTH {JUMLAH_BOOTH}x({BOOTH_LAYOUTS[idx_booth_layout]})(L) | "
                      f"FPS {int(fps)} | FOTO {jml_foto}",
             (22, h - 24), 0.55, 2)

    if mode == "LENSA" and quad is not None and hilang < 6:
        r, p, y = orientasi(quad.astype(np.float32))
        teks(tampil, f"ROLL {r:+.0f}  PITCH {p:+.0f}  YAW {y:+.0f}",
             (w - 320, h - 24), 0.55, 2, AMBER)

    if debug:
        b = " ".join("".join("1" if x else "0" for x in f) for _, f, _ in info)
        teks(tampil, f"TANGAN {len(info)}  JARI[{b}]  {status}",
             (22, h - 92), 0.5, 2, AMBER)

    # --- Terapkan zoom ke frame akhir sebelum ditampilkan ---
    if zoom_aktif:
        cv2.drawMarker(tampil, titik_int(zoom_pusat), CYAN,
                       markerType=cv2.MARKER_CROSS, markerSize=22, thickness=1)
    tampil = terapkan_zoom(tampil, zoom_level, tuple(zoom_pusat))
    if zoom_aktif or zoom_level > 1.02:
        teks(tampil, f"ZOOM {zoom_level:.1f}x", (w - 170, 78), 0.6, 2,
             CYAN if zoom_aktif else ABU)

    if now < kilat_sampai:
        a = (kilat_sampai - now) / 0.30
        tampil = cv2.addWeighted(tampil, 1 - a, np.full_like(tampil, 255), a, 0)

    # --- Thumbnail foto terakhir ---
    if thumbnail_terakhir is not None and (now - thumbnail_waktu) < THUMB_TAHAN:
        th_img = cv2.resize(thumbnail_terakhir, (THUMB_W, THUMB_H))
        tx, ty = w - THUMB_W - 18, 54
        tampil[ty:ty + THUMB_H, tx:tx + THUMB_W] = th_img
        cv2.rectangle(tampil, (tx - 2, ty - 2), (tx + THUMB_W + 2, ty + THUMB_H + 2),
                      PUTIH, 2)
        teks(tampil, "TERAKHIR", (tx, ty - 8), 0.42, 1, PUTIH, aberasi=False)

    # --- Panel bantuan gestur (2 Kolom Kartu Visual Legibel) ---
    if bantuan:
        ov = tampil.copy()
        cv2.rectangle(ov, (30, 25), (w - 30, h - 25), (16, 14, 22), cv2.FILLED)
        tampil = cv2.addWeighted(ov, 0.90, tampil, 0.10, 0)
        cv2.rectangle(tampil, (30, 25), (w - 30, h - 25), CYAN, 2, cv2.LINE_AA)
        cv2.rectangle(tampil, (33, 28), (w - 33, h - 28), MAGENTA, 1, cv2.LINE_AA)

        teks(tampil, "PANDUAN GESTUR HAND & KONTROL RETROLENS", (50, 58), 0.68, 2, CYAN)

        # Column 1: Gestures
        col1_x1, col1_y1, col1_x2, col1_y2 = 45, 75, 465, h - 35
        cv2.rectangle(tampil, (col1_x1, col1_y1), (col1_x2, col1_y2), (26, 22, 34), cv2.FILLED)
        cv2.rectangle(tampil, (col1_x1, col1_y1), (col1_x2, col1_y2), MAGENTA, 1)
        teks(tampil, "HAND GESTURES", (col1_x1 + 15, col1_y1 + 24), 0.52, 2, MAGENTA, aberasi=False)

        g_list = [
            ("Peace (v) Tahan 3dtk", "Konfirmasi Foto Efek, lalu turunkan tangan (hitung mundur jalan sendiri)"),
            ("2 Tangan (L)", "Buka & Kunci Bingkai Lensa Virtual"),
            ("1 Tangan (L)", "Zoom Kamera (Cubit Jempol-Telunjuk)"),
            ("Telunjuk Tahan", "Masuk Mode Gambar / Coret Kanvas"),
            ("3 Jari Tahan", "Mode Blur + Putar Musik Relaksasi"),
            ("Telapak Terbuka", "Kembali ke Mode Lensa Utama"),
            ("Kepal Tangan", "Hapus Seluruh Coretan Kanvas"),
        ]
        for idx, (title, desc) in enumerate(g_list):
            y_curr = col1_y1 + 52 + idx * 50
            cv2.rectangle(tampil, (col1_x1 + 10, y_curr - 18), (col1_x2 - 10, y_curr + 24), (36, 30, 48), cv2.FILLED)
            cv2.rectangle(tampil, (col1_x1 + 10, y_curr - 18), (col1_x2 - 10, y_curr + 24), SAMAR, 1)
            teks(tampil, title, (col1_x1 + 18, y_curr), 0.44, 2, AMBER, aberasi=False)
            teks(tampil, desc, (col1_x1 + 18, y_curr + 18), 0.38, 1, PUTIH, aberasi=False)

        # Column 2: Keyboard Hotkeys
        col2_x1, col2_y1, col2_x2, col2_y2 = 485, 75, w - 45, h - 35
        cv2.rectangle(tampil, (col2_x1, col2_y1), (col2_x2, col2_y2), (26, 22, 34), cv2.FILLED)
        cv2.rectangle(tampil, (col2_x1, col2_y1), (col2_x2, col2_y2), CYAN, 1)
        teks(tampil, "KEYBOARD SHORTCUTS", (col2_x1 + 15, col2_y1 + 24), 0.52, 2, CYAN, aberasi=False)

        k_list = [
            ("[ F ]", "Buka Menu Pemilih Bingkai (Frame Picker)"),
            ("[ T ]", "Mulai Sesi PHOTOBOOTH Otomatis"),
            ("[ L ]", "Ubah Layout Booth (Vertikal/Grid/Horiz)"),
            ("[ [ / ] ]", "Atur Jumlah Foto Photobooth (2-6 Foto)"),
            ("[ B / M ]", "Toggle Mode Blur / Switch Mode Gambar"),
            ("[ E / P ]", "Ganti Efek Foto / Ganti Warna Pena"),
            ("[ C / U ]", "Hapus Kanvas / Undo Coretan"),
            ("[ S / R ]", "Simpan Foto Manual / Reset Zoom 1.0x"),
            ("[ H / Q ]", "Tutup Bantuan (H) / Keluar Program (Q)"),
        ]
        for idx, (key, desc) in enumerate(k_list):
            y_curr = col2_y1 + 52 + idx * 39
            cv2.rectangle(tampil, (col2_x1 + 10, y_curr - 16), (col2_x2 - 10, y_curr + 18), (36, 30, 48), cv2.FILLED)
            teks(tampil, key, (col2_x1 + 18, y_curr + 2), 0.44, 2, CYAN, aberasi=False)
            teks(tampil, desc, (col2_x1 + 115, y_curr + 2), 0.38, 1, PUTIH, aberasi=False)

    # --- Frame Picker Menu Modal Overlay ---
    tampil = gambar_menu_pemilih_bingkai(tampil, idx_bingkai, menu_bingkai_terbuka)

    # --- Bungkus tampilan akhir dengan bingkai tema ---
    tampil_layar = tampilkan_dengan_bingkai(tampil, idx_bingkai)
    cv2.imshow("RETROLENS", tampil_layar)

    k = cv2.waitKey(1) & 0xFF
    if k == ord("q"):
        break
    elif k == ord("f"):
        menu_bingkai_terbuka = not menu_bingkai_terbuka
    elif k == ord("l"):
        idx_booth_layout = (idx_booth_layout + 1) % len(BOOTH_LAYOUTS)
    elif k == ord("[") or k == ord("a") or k == 81:
        if menu_bingkai_terbuka:
            idx_bingkai = (idx_bingkai - 1) % len(BINGKAI_LUCU)
        else:
            JUMLAH_BOOTH = max(BOOTH_MIN, JUMLAH_BOOTH - 1)
    elif k == ord("]") or k == ord("d") or k == 83:
        if menu_bingkai_terbuka:
            idx_bingkai = (idx_bingkai + 1) % len(BINGKAI_LUCU)
        else:
            JUMLAH_BOOTH = min(BOOTH_MAKS, JUMLAH_BOOTH + 1)
    elif k == 13:  # Enter
        if menu_bingkai_terbuka:
            menu_bingkai_terbuka = False
    elif k == ord(" "):
        if menu_bingkai_terbuka:
            menu_bingkai_terbuka = False
        else:
            idx_lensa = (idx_lensa + 1) % len(LENSA)
    elif k == ord("e"):
        idx_efek_peace = (idx_efek_peace + 1) % len(EFEK_PEACE)
    elif k == ord("t"):
        if mode == "LENSA" and not booth_aktif:
            booth_aktif = True
            booth_daftar = []
            booth_hitung_mulai = None
    elif k == 27:  # ESC
        if menu_bingkai_terbuka:
            menu_bingkai_terbuka = False
        elif booth_aktif:
            booth_aktif = False
            booth_daftar = []
            booth_hitung_mulai = None
    elif k == ord("m"):
        if mode == "BLUR":
            musik_hentikan()
        old_mode = mode
        mode = "GAMBAR" if mode == "LENSA" else "LENSA"
        mulai_diam = mulai_ganti = quad = None
        pena_akhir = pena_halus = None
        menggambar = False
        beruntun_ya = beruntun_tidak = 0
        hilang = 99
        blur_mode_kuat = 1.0
        blur_mode_target = 1.0
        blur_peace_absen = 0
        blur_peace_terlihat = False
    elif k == ord("b"):
        if mode == "BLUR":
            musik_hentikan()
            mode = "LENSA"
            mulai_ganti = None
            quad = None
            hilang = 99
            blur_mode_kuat = 1.0
            blur_mode_target = 1.0
            blur_peace_absen = 0
            blur_peace_terlihat = False
        else:
            mode = "BLUR"
            mulai_diam = mulai_ganti = quad = None
            blur_mode_kuat = 1.0
            blur_mode_target = 1.0
            blur_peace_absen = 0
            blur_peace_terlihat = False
            musik_putar()
    elif k == ord("c"):
        kanvas[:] = 0
        strokes.clear()
        goresan = 0
    elif k == ord("u"):
        if strokes:
            strokes.pop()
            gambar_ulang_kanvas()
            goresan = sum(len(s["titik"]) for s in strokes)
    elif k == ord("p"):
        idx_pena = (idx_pena + 1) % len(PENA)
    elif k == ord("s"):
        simpan(tampil, "MANUAL", pakai_bingkai=False)
    elif k == ord("d"):
        debug = not debug
    elif k == ord("r"):
        zoom_target = zoom_level = 1.0
    elif k == ord("h"):
        bantuan = not bantuan

musik_hentikan()
kamera.stop()
cv2.destroyAllWindows()
landmarker.close()
pygame.mixer.quit()
print(f"Selesai. {jml_foto} foto tersimpan di folder '{FOLDER}'.")