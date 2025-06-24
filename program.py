import numpy as np
from PIL import Image
import io
import base64
import matplotlib.pyplot as plt

def jalanPCA(x, jml):
    avg = np.mean(x, axis=0)
    cen = x - avg

    U, S, Vt = np.linalg.svd(cen, full_matrices=False)

    komponen = Vt[:jml] 

    proj = np.dot(cen, komponen.T) 

    balik = np.dot(proj, komponen) + avg

    var_lengkap = (S ** 2) / (x.shape[0] - 1)
    tot = np.sum(var_lengkap)
    ex = np.sum(var_lengkap[:jml]) / tot if tot > 0 else 0

    return np.clip(balik, 0, 255), ex, var_lengkap[:jml]

def ambilJml(dim, qual):
    if qual <= 20:
        ras = 0.02
    elif qual <= 40:
        ras = 0.08
    elif qual <= 60:
        ras = 0.2
    elif qual <= 80:
        ras = 0.4
    else:
        ras = 0.6
    jml = max(1, int(dim * ras))
    return min(jml, dim)

def ambilJpgQ(qual):
    if qual <= 20:
        return 25
    elif qual <= 40:
        return 40
    elif qual <= 60:
        return 60
    elif qual <= 80:
        return 75
    else:
        return 85

def ukurStr(s):
    if s < 1024: return f"{s} B"
    elif s < 1024 * 1024: return f"{s / 1024:.1f} KB"
    else: return f"{s / (1024 * 1024):.1f} MB"

def cepetPCA(gam, qual, maxkb, maxdim):
    ori = gam.size

    if qual <= 30:
        sc = 0.6
    elif qual <= 50:
        sc = 0.8
    else:
        sc = 1.0

    dim = int(maxdim * sc)

    if max(gam.size) > dim:
        gam.thumbnail((dim, dim), Image.Resampling.LANCZOS)

    arr = np.array(gam)
    h, w = arr.shape[:2]
    komp = min(h, w)
    pakai = ambilJml(komp, qual)

    hasil = np.zeros_like(arr, dtype=np.uint8)
    semuaVar = []
    eigRed = None

    for ch in range(3):
        dat = arr[:, :, ch].astype(np.float64)
        rek, var, eigs = jalanPCA(dat, pakai)
        hasil[:, :, ch] = rek.astype(np.uint8)
        semuaVar.append(var)
        if ch == 0:
            eigRed = eigs

    jadi = Image.fromarray(hasil)
    jpgq = ambilJpgQ(qual)

    f = "JPEG"
    buf = None
    best = float('inf')

    jpgbuf = io.BytesIO()
    jadi.save(jpgbuf, format="JPEG", quality=jpgq, optimize=True)
    jpgsiz = len(jpgbuf.getvalue())
    f = "JPEG"
    buf = jpgbuf.getvalue()
    best = jpgsiz

    pngbuf = io.BytesIO()
    jadi.save(pngbuf, format="PNG", optimize=True)
    pngsiz = len(pngbuf.getvalue())
    if pngsiz < best:
        f = "PNG"
        buf = pngbuf.getvalue()
        best = pngsiz

    if best / 1024 > maxkb and qual > 20:
        for q in [jpgq - 15, jpgq - 25, 20, 15]:
            if q > 10:
                test = io.BytesIO()
                jadi.save(test, format="JPEG", quality=q, optimize=True)
                sz = len(test.getvalue())
                if sz < best:
                    f = "JPEG"
                    buf = test.getvalue()
                    best = sz
                if sz / 1024 <= maxkb:
                    break

    asli = h * w * 3
    kompr = (h * pakai + pakai * w) * 3
    red = 100 - (kompr / asli * 100)

    return {
        'img': hasil,
        'var': semuaVar,
        'eig': eigRed,
        'cmp': pakai,
        'byt': best,
        'kb': best / 1024,
        'fmt': f,
        'buf': buf,
        'red': red,
        'ori': ori,
        'fin': (w, h)
    }

def fig2b64(fig):
    b = io.BytesIO()
    fig.savefig(b, format='png', bbox_inches='tight', dpi=80, facecolor='#1e293b', edgecolor='none')
    b.seek(0)
    return base64.b64encode(b.read()).decode('utf-8')