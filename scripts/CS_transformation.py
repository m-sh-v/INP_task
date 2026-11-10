import pandas as pd
import numpy as np
from pathlib import Path
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot


#* ===== Загрузка данных =====

PROJECT_PATH = Path(__file__).parent.parent
DATA_FOLDER = PROJECT_PATH / 'data_files'

mf_csv = DATA_FOLDER / 'cleaning' / "enc_data_full.csv"
tracker_csv = DATA_FOLDER / 'cleaning' / "laser_data_full.csv"


enc = pd.read_csv(mf_csv)           # encoder
enc["Time"] = pd.to_datetime(enc["Time"])

las = pd.read_csv(tracker_csv)      # laser
las["time_laser"] = pd.to_datetime(
    las["time_laser"].str.strip(),
    format="%m/%d/%y %H:%M:%S"
)

print(enc.shape, las.shape)
print(enc.head(3))
print(las.head(3))


print("Прочитали файлы")


#* ===== Начальная визуализация =====

def plot_2d_and_3d(A, B, labelA="encoder", labelB="laser",
                   colsA=("X","Y","Z"), colsB=("X_axis","Y_axis","Z_axis")):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    pairs = [(0,1), (0,2), (1,2)]
    names = ["XY", "XZ", "YZ"]
    for ax, (i,j), nm in zip(axes, pairs, names):
        ax.plot(A[colsA[i]], A[colsA[j]], '.', ms=2, alpha=.6, label=labelA)
        ax.plot(B[colsB[i]], B[colsB[j]], '.', ms=2, alpha=.6, label=labelB)
        ax.set_xlabel(colsA[i][:1] + " / " + colsB[i][:1])
        ax.set_ylabel(colsA[j][:1] + " / " + colsB[j][:1])
        ax.set_title(nm); ax.legend(); ax.set_aspect("equal", "datalim")
    plt.tight_layout(); plt.show()

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(A[colsA[0]], A[colsA[1]], A[colsA[2]], ".", ms=2, alpha=.6, label=labelA)
    ax.plot(B[colsB[0]], B[colsB[1]], B[colsB[2]], ".", ms=2, alpha=.6, label=labelB)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.legend(); plt.tight_layout(); plt.show()



K_FI = 2 * np.pi / 50000
K_RZ = 120.0 / 100000

enc["Fi_rad"] = enc["Position Fi"] * K_FI
enc["R_mm"] = enc["Position R"] * K_RZ
enc["Z_mm"] = enc["Position Z"] * K_RZ

def enc_to_cart(df, Fi0=0.0, R0=0.0, Z0=0.0):       # Из цилиндра в декарта
    Fi = df["Fi_rad"].values + Fi0
    R  = df["R_mm"].values + R0
    Z  = df["Z_mm"].values + Z0
    return np.column_stack([R * np.cos(Fi), R * np.sin(Fi), Z])

enc_xyz = enc_to_cart(enc)
enc[["X_enc", "Y_enc", "Z_enc"]] = enc_xyz
las_xyz = las[["X_axis", "Y_axis", "Z_axis"]].values


plot_2d_and_3d(enc, las,
               colsA=("X_enc", "Y_enc", "Z_enc"),
               colsB=("X_axis", "Y_axis", "Z_axis"))


def kabsch(P, Q):
    cp, cq = P.mean(0), Q.mean(0)
    H = (P - cp).T @ (Q - cq)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    t = cq - R @ cp
    return R, t


# Попробуем грубо сопоставить 
enc_t = enc["Time"].values.astype("datetime64[s]").astype(np.int64)
las_t = las["time_laser"].values.astype("datetime64[s]").astype(np.int64)

t_offset = float(np.median(enc_t) - np.median(las_t))
print(f"Медианный сдвиг времени enc−laser: {t_offset:.2f} с")

tree = cKDTree((las_t + t_offset).reshape(-1, 1))       # idx_t = для каждого момента энкодера ближайший момент лазера
dist_t, idx_t = tree.query(enc_t.reshape(-1, 1))        # dist_t = расстояние по времени

TOL_T = 10.0
keep = dist_t < TOL_T       # Будем оставлять пары с порогом 10
matches = np.column_stack([np.where(keep)[0], idx_t[keep]])
print(f"Грубых соответствий по времени: {len(matches)}")


# Начальное приближение
P0 = enc_xyz[matches[:, 0]]
Q0 = las_xyz[matches[:, 1]]
R0_kabsch, t0_kabsch = kabsch(P0, Q0)
r0_euler = Rot.from_matrix(R0_kabsch).as_euler("xyz")
print("Kabsch initial RMS, мм:",
      np.sqrt(np.mean(np.linalg.norm(P0 @ R0_kabsch.T + t0_kabsch - Q0, axis=1) ** 2)))
print("Euler (xyz), рад:", r0_euler, "  t, мм:", t0_kabsch)


def residuals(params, enc_df, las_xyz, matches):
    Fi0, R0, Z0, ax, ay, az, tx, ty, tz = params
    Fi = enc_df["Fi_rad"].values + Fi0
    R  = enc_df["R_mm"].values   + R0
    Z  = enc_df["Z_mm"].values   + Z0
    P  = np.column_stack([R * np.cos(Fi), R * np.sin(Fi), Z])
    Rm = Rot.from_euler("xyz", [ax, ay, az]).as_matrix()
    P_tr = P @ Rm.T + np.array([tx, ty, tz])
    i_enc, i_las = matches[:, 0], matches[:, 1]
    return (P_tr[i_enc] - las_xyz[i_las]).ravel()

x0 = np.array([0.0, 0.0, 0.0, *r0_euler, *t0_kabsch])

sol = least_squares(
    residuals, x0, args=(enc, las_xyz, matches),
    loss="soft_l1",
    f_scale=20.0,        
    verbose=2, 
    max_nfev=2000,
)
print("Параметры (грубо):", sol.x)
print("RMS невязки, мм:", np.sqrt(np.mean(sol.fun ** 2)))



def icp_refine(enc_df, las_xyz, x0, n_iter=20, reject_mm=20.0, f_scale=1.0):
    x = x0.copy()
    tree = cKDTree(las_xyz)
    for it in range(n_iter):
        Fi = enc_df["Fi_rad"].values + x[0]
        R  = enc_df["R_mm"].values + x[1]
        Z  = enc_df["Z_mm"].values + x[2]
        P  = np.column_stack([R * np.cos(Fi), R * np.sin(Fi), Z])
        Rm = Rot.from_euler("xyz", x[3:6]).as_matrix()
        P_tr = P @ Rm.T + x[6:9]

        dist, idx = tree.query(P_tr)
        keep = dist < reject_mm             # Оставляем пары, у которых расстояние меньше 20 мм
        matches = np.column_stack([np.where(keep)[0], idx[keep]])
        if len(matches) < 10:
            print("Слишком мало пар — останов ICP")
            break

        sol = least_squares(residuals, x,
                            args=(enc_df, las_xyz, matches),
                            loss="soft_l1", f_scale=f_scale)
        x = sol.x
        print(f"iter {it:02d}: RMS={np.sqrt(np.mean(sol.fun ** 2)):7.3f} мм, "
              f"пар={len(matches)}")
    return x, matches


x_final, matches_final = icp_refine(enc, las_xyz, sol.x, n_iter=20, f_scale=1.0)


#* ===== Применяем модель и визуализируем =====

def apply_model(enc_df, x):
    Fi0, R0, Z0, ax, ay, az, tx, ty, tz = x
    Fi = enc_df["Fi_rad"].values + Fi0
    R  = enc_df["R_mm"].values + R0
    Z  = enc_df["Z_mm"].values + Z0
    P  = np.column_stack([R * np.cos(Fi), R * np.sin(Fi), Z])
    Rm = Rot.from_euler("xyz", [ax, ay, az]).as_matrix()
    return P @ Rm.T + np.array([tx, ty, tz])

enc_aligned = apply_model(enc, x_final)         # Выровненные координаты
enc["X_al"], enc["Y_al"], enc["Z_al"] = enc_aligned.T

plot_2d_and_3d(enc, las,
               colsA=("X_al", "Y_al", "Z_al"),
               colsB=("X_axis", "Y_axis", "Z_axis"))

dist, _ = cKDTree(las_xyz).query(enc_aligned)
plt.hist(dist, bins=50)
plt.xlabel("Расстояние до ближайшей лазерной точки, мм")
plt.ylabel("Кол-во")
plt.title("Гистограмма остатков")           # По идее, должно быть узко вокруг 0
plt.show()

