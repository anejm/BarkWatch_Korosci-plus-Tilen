"""
ARSO – prenos vseh postaj in dnevnih podatkov (2000–2026)
=========================================================
Poženi: pip install requests
Nato:   python arso_prenos.py

Ustvari dve datoteki:
  - arso_postaje.csv        (id, ime, lon, lat, alt, tip)
  - arso_dnevni_podatki.csv (station_id, datum, + dinamični stolpci po postaji)

Upošteva:
  - limit 1000 dni na poizvedbo → chunki po 900 dni
  - počasne zahteve (delay) da ne preobremenimo strežnika
  - dinamični stolpci – vsaka postaja ima le tiste vars ki jih ima
"""

import requests
import csv
import re
import random
import string
import datetime
import time

# ── Nastavitve ───────────────────────────────────────────────
D1            = datetime.date(2000, 1, 1)
D2            = datetime.date(2026, 3, 20)
CHUNK_DAYS    = 900
VARS          = "35,46,38,50,36,51,85,52,88,62,89,81,41"
WORKERS       = 3
DELAY_BETWEEN = 1.0
IZHOD_META    = "arso_postaje.csv"
IZHOD_DATA    = "arso_dnevni_podatki.csv"

LOCATIONS_URL = (
    "https://meteo.arso.gov.si/webmet/archive/locations.xml"
    "?d1=2016-03-24&d2=2026-03-12&type=3,2,1&%20lang=si&nocache=mn0ax8tw2ivyqe89rts"
)
DATA_URL = "https://meteo.arso.gov.si/webmet/archive/data.xml"
EPOCH    = datetime.datetime(1800, 1, 1)
# ─────────────────────────────────────────────────────────────

def nocache():
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=16))

def ts_to_date(ts_str):
    return (EPOCH + datetime.timedelta(minutes=int(ts_str))).strftime("%Y-%m-%d")

def date_chunks(d1, d2, chunk_days):
    chunks = []
    cur = d1
    while cur <= d2:
        end = min(cur + datetime.timedelta(days=chunk_days - 1), d2)
        chunks.append((cur, end))
        cur = end + datetime.timedelta(days=1)
    return chunks

# ── 1. Preberi postaje iz locations.xml ─────────────────────

def preberi_postaje():
    print("Prenašam seznam postaj iz locations.xml ...")
    r = requests.get(LOCATIONS_URL, timeout=20)
    text = r.text
    pattern = re.compile(
        r'_(\d+):\{\s*name:"([^"]+)",\s*lon:([\d.\-]+),\s*lat:([\d.\-]+),\s*alt:([\d.\-]+),\s*type:(\d+)'
    )
    postaje = []
    for m in pattern.finditer(text):
        postaje.append({
            "id":  int(m.group(1)),
            "ime": m.group(2),
            "lon": float(m.group(3)),
            "lat": float(m.group(4)),
            "alt": int(float(m.group(5))),
            "tip": int(m.group(6)),
        })
    postaje.sort(key=lambda x: x["id"])
    print(f"  Najdenih {len(postaje)} postaj.")
    return postaje

# ── 2. Parsiraj en XML odgovor ───────────────────────────────

def parsiraj_odgovor(text, station_id):
    m = re.search(r'points:\{(.+)\}\s*\)\s*\]\]', text, re.DOTALL)
    if not m:
        return {}, []

    points_str = m.group(1)

    if not re.search(rf'\b_{station_id}\s*:', points_str):
        return {}, []

    # Izvleci params: p0 -> kratko ime (s), p1 -> ... ter pid za referenco
    # Format: pN:{ pid:"XX", name:"...", s:"kratko ime", l:"dolgo ime", unit:"enota"}
    stolpci = {}  # pk -> "ime_enota"
    for pm in re.finditer(
        r'(p\d+):\{\s*pid:"(\d+)",\s*name:"([^"]+)",\s*s:"([^"]+)",\s*l:"([^"]*)",\s*unit:"([^"]*)"',
        text
    ):
        pk      = pm.group(1)
        pid     = pm.group(2)
        s_ime   = pm.group(4)   # kratko ime
        unit    = pm.group(6)   # enota

        # Sestavi ime stolpca: kratko_ime (enota) → počisti za CSV header
        if unit:
            col = f"{s_ime}_{unit}"
        else:
            col = s_ime
        # Počisti znake ki niso primerni za CSV header
        col = col.replace(" ", "_").replace(".", "").replace("/", "_") \
                 .replace("(", "").replace(")", "").replace("°", "deg") \
                 .replace("%", "pct").replace(",", "").replace("²", "2") \
                 .replace(">", "gt").replace("<", "lt").replace("=", "")
        stolpci[pk] = col

    if not stolpci:
        return {}, []

    # Izvleci vse dneve
    vrstice = []
    for dm in re.finditer(r'_(\d+):\{([^}]+)\}', points_str):
        ts       = dm.group(1)
        vals_str = dm.group(2)

        if ':"' not in vals_str:
            continue

        try:
            datum = ts_to_date(ts)
        except Exception:
            continue

        vrstica = {"datum": datum}
        for pk in stolpci:
            v = re.search(rf'{pk}:"([^"]*)"', vals_str)
            vrstica[pk] = v.group(1) if v else ""

        vrstice.append(vrstica)

    return stolpci, vrstice

# ── 3. Prenesi vse chunke za eno postajo ────────────────────

def prenesi_postajo(station_id, chunks, delay):
    vse_stolpci = {}
    vse_vrstice = []

    for i, (c_d1, c_d2) in enumerate(chunks):
        params = {
            "lang":    "si",
            "vars":    VARS,
            "group":   "dailyData0",
            "type":    "daily",
            "id":      str(station_id),
            "d1":      c_d1.strftime("%Y-%m-%d"),
            "d2":      c_d2.strftime("%Y-%m-%d"),
            "nocache": nocache(),
        }
        try:
            r = requests.get(DATA_URL, params=params, timeout=30)
            stolpci, vrstice = parsiraj_odgovor(r.text, station_id)
            if stolpci and not vse_stolpci:
                vse_stolpci = stolpci
            vse_vrstice.extend(vrstice)
        except Exception as e:
            print(f"    ✗ ID {station_id} chunk {i+1}: {e}")

        if i < len(chunks) - 1:
            time.sleep(delay)

    return station_id, vse_stolpci, vse_vrstice

# ── 4. Glavna logika ─────────────────────────────────────────

def main():
    postaje = preberi_postaje()

    with open(IZHOD_META, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id","ime","lon","lat","alt","tip"])
        writer.writeheader()
        writer.writerows(postaje)
    print(f"  → Shranjeno: {IZHOD_META}\n")

    chunks = date_chunks(D1, D2, CHUNK_DAYS)
    print(f"Obdobje {D1} – {D2} razdeljeno na {len(chunks)} chunkov po ~{CHUNK_DAYS} dni.")
    print(f"Postaj: {len(postaje)}, workers: {WORKERS}, delay: {DELAY_BETWEEN}s")
    print(f"Vars: {VARS}")
    print(f"Skupaj zahtev: ~{len(postaje) * len(chunks)}")
    print("-" * 65)

    from concurrent.futures import ThreadPoolExecutor, as_completed

    vse_stolpci_global = {}
    rezultati = {}
    skupaj  = len(postaje)
    stevec  = 0
    zacetek = time.time()

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {
            pool.submit(prenesi_postajo, p["id"], chunks, DELAY_BETWEEN): p["id"]
            for p in postaje
        }
        for fut in as_completed(futures):
            sid, stolpci, vrstice = fut.result()
            stevec += 1

            if stolpci:
                for pk, ime in stolpci.items():
                    if pk not in vse_stolpci_global:
                        vse_stolpci_global[pk] = ime
                rezultati[sid] = vrstice

                elapsed = time.time() - zacetek
                rate    = stevec / elapsed
                eta     = int((skupaj - stevec) / rate) if rate > 0 else 0
                print(f"  ✓ ID {sid:5d} | {len(vrstice):6d} dni | "
                      f"[{stevec}/{skupaj}] ETA: {eta}s")
            else:
                print(f"  - ID {sid:5d} | brez podatkov [{stevec}/{skupaj}]")

    # Sortirani stolpci po številki (p0, p1, p2 ...)
    pk_sorted  = sorted(vse_stolpci_global.keys(), key=lambda x: int(x[1:]))
    col_names  = [vse_stolpci_global[pk] for pk in pk_sorted]
    fieldnames = ["station_id", "datum"] + col_names

    # Izpiši kateri stolpci so bili najdeni
    print(f"\nNajdeni stolpci: {col_names}")

    skupaj_vrstic = 0
    with open(IZHOD_DATA, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

        for p in sorted(postaje, key=lambda x: x["id"]):
            sid = p["id"]
            if sid not in rezultati:
                continue
            for vrstica in sorted(rezultati[sid], key=lambda r: r["datum"]):
                row = {"station_id": sid, "datum": vrstica["datum"]}
                for pk in pk_sorted:
                    col_ime = vse_stolpci_global[pk]
                    row[col_ime] = vrstica.get(pk, "")
                writer.writerow(row)
                skupaj_vrstic += 1

    elapsed = time.time() - zacetek
    print("-" * 65)
    print(f"Končano v {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Skupaj vrstic: {skupaj_vrstic:,}")
    print(f"  → {IZHOD_META}")
    print(f"  → {IZHOD_DATA}")

if __name__ == "__main__":
    main()