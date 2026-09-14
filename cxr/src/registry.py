"""
Auto-discovery of chest X-ray domains from whatever is attached on Kaggle.

WHY: Kaggle dataset slugs and folder layouts change. Rather than hardcode
paths that break, we scan /kaggle/input for known LAYOUTS and build the
domain list from what is actually there.

Each recipe recognises one public dataset by its directory signature and
emits one or more DOMAIN SPECS. A spec is fully resolved -- concrete paths,
no guessing left for later.

Supported datasets (search these on Kaggle):
  1. "Chest X-Ray Images (Pneumonia)"        paultimothymooney/chest-xray-pneumonia
  2. "COVID-19 Radiography Database"         tawsifurrahman/covid19-radiography-database
  3. "Tuberculosis (TB) Chest X-ray Database" tawsifurrahman/tuberculosis-tb-chest-xray-dataset
  4. "Pulmonary Chest X-Ray Abnormalities"   kmader/pulmonary-chest-xray-abnormalities
  5. "NIH Chest X-rays" (sample or full)     nih-chest-xrays/sample  |  nih-chest-xrays/data
  6. "Chest X-rays Indiana University"       raddar/chest-xrays-indiana-university
"""
from pathlib import Path

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp"}


def _dirs_with_images(p, min_n=20):
    """Does this directory hold at least min_n images (directly)?"""
    if not p.is_dir():
        return False
    n = 0
    for f in p.iterdir():
        if f.suffix.lower() in IMG_EXT:
            n += 1
            if n >= min_n:
                return True
    return False


def _image_dir(base, cls):
    """
    Class folders sometimes hold images directly, sometimes inside images/.
    Return whichever actually contains the files.
    """
    for cand in (base / cls / "images", base / cls):
        if _dirs_with_images(cand):
            return cand
    return None


# ---------------------------------------------------------------- recipe 1
def recipe_kermany(root):
    """chest_xray/{train,test,val}/{NORMAL,PNEUMONIA}"""
    out = []
    for d in root.rglob("NORMAL"):
        if d.parent.name.lower() not in ("train", "test", "val"):
            continue
        base = d.parent.parent
        norm = [p for p in base.glob("*/NORMAL") if _dirs_with_images(p)]
        abn = [p for p in base.glob("*/PNEUMONIA") if _dirs_with_images(p)]
        if norm and abn:
            out.append(dict(
                name="kermany_peds", mode="globs",
                normal_dirs=norm, abnormal_dirs=abn,
                meta=dict(country="China", city="Guangzhou",
                          population="paediatric", finding="pneumonia",
                          source="Kermany 2018")))
            break
    return out


# ---------------------------------------------------------------- recipe 2
def recipe_covid_radiography(root):
    """
    COVID-19_Radiography_Dataset/{Normal,COVID,Lung_Opacity,Viral Pneumonia}

    All three abnormal classes share ONE Normal pool. Using the same normal
    images in several 'domains' would be leakage, so the pool is split into
    disjoint thirds via the `partition` field.
    """
    out = []
    for base in list(root.rglob("Normal")) + [root]:
        base = base.parent if base.name == "Normal" else base
        nd = _image_dir(base, "Normal")
        if nd is None:
            continue
        classes = [("Viral Pneumonia", "covid_viral", "viral pneumonia"),
                   ("Lung_Opacity", "covid_opacity", "lung opacity"),
                   ("COVID", "covid_covid", "covid-19")]
        found = [(c, n, f) for c, n, f in classes if _image_dir(base, c)]
        if not found:
            continue
        for i, (cls, name, finding) in enumerate(found):
            out.append(dict(
                name=name, mode="globs",
                normal_dirs=[nd], abnormal_dirs=[_image_dir(base, cls)],
                partition=(i, len(found)),          # disjoint slice of Normal
                meta=dict(country="Qatar/Bangladesh", population="adult",
                          finding=finding, source="COVID-19 Radiography DB")))
        break
    return out


# ---------------------------------------------------------------- recipe 3
def recipe_tb_qatar(root):
    """TB_Chest_Radiography_Database/{Normal,Tuberculosis}"""
    out = []
    for d in root.rglob("Tuberculosis"):
        base = d.parent
        nd, ad = _image_dir(base, "Normal"), _image_dir(base, "Tuberculosis")
        if nd and ad:
            out.append(dict(
                name="tb_qatar", mode="globs",
                normal_dirs=[nd], abnormal_dirs=[ad],
                meta=dict(country="Qatar/Bangladesh", population="adult",
                          finding="tuberculosis", source="TB Chest Radiography DB")))
            break
    return out


# ---------------------------------------------------------------- recipe 4
def recipe_shenzhen_montgomery(root):
    """
    ChinaSet_AllFiles/CXR_png  and  MontgomerySet/CXR_png
    Label is the last character of the filename stem: _0 normal, _1 abnormal.
    Two genuinely different countries -- the cleanest domain pair available.
    """
    out = []
    for tag, name, meta in (
        ("ChinaSet", "shenzhen", dict(country="China", city="Shenzhen")),
        ("Montgomery", "montgomery", dict(country="USA", city="Montgomery County")),
    ):
        for d in root.rglob("CXR_png"):
            if tag.lower() in str(d).lower() and _dirs_with_images(d, 50):
                out.append(dict(
                    name=name, mode="filename", dirs=[d],
                    suffix_map={"0": 0, "1": 1},
                    meta=dict(population="adult", finding="tuberculosis",
                              source="NLM TB set", **meta)))
                break
    return out


# ---------------------------------------------------------------- recipe 5
def recipe_nih(root):
    """
    NIH ChestX-ray14, sample or full.
    Split into TWO domains by View Position: PA (standing) vs AP (supine,
    typically sicker patients). Different geometry and case mix -- a real
    domain shift, and free.
    """
    csv = None
    for pat in ("sample_labels.csv", "Data_Entry_2017*.csv"):
        hits = list(root.rglob(pat))
        if hits:
            csv = hits[0]
            break
    if csv is None:
        return []
    img_dirs = [d for d in root.rglob("images") if _dirs_with_images(d, 50)]
    if not img_dirs:
        return []
    out = []
    for view in ("PA", "AP"):
        out.append(dict(
            name=f"nih_{view.lower()}", mode="csv", csv=csv,
            image_col="Image Index", label_col="Finding Labels",
            normal_values=["No Finding"],
            filter={"View Position": [view]},
            image_dirs=img_dirs,
            meta=dict(country="USA", population="adult", view=view,
                      finding="any of 14", source="NIH ChestX-ray14")))
    return out


# ---------------------------------------------------------------- recipe 6
def recipe_indiana(root):
    """Indiana / Open-i. Labels come from the reports table."""
    proj = list(root.rglob("indiana_projections.csv"))
    rep = list(root.rglob("indiana_reports.csv"))
    if not (proj and rep):
        return []
    img_dirs = [d for d in root.rglob("images_normalized") if _dirs_with_images(d, 50)]
    if not img_dirs:
        img_dirs = [d for d in root.rglob("images") if _dirs_with_images(d, 50)]
    if not img_dirs:
        return []
    return [dict(name="indiana", mode="indiana", projections=proj[0],
                 reports=rep[0], image_dirs=img_dirs,
                 meta=dict(country="USA", state="Indiana", population="adult",
                           finding="any", source="Open-i / Indiana"))]


RECIPES = [recipe_kermany, recipe_covid_radiography, recipe_tb_qatar,
           recipe_shenzhen_montgomery, recipe_nih, recipe_indiana]


def discover(input_root="/kaggle/input", verbose=True):
    """Scan every attached dataset and return all domain specs found."""
    root = Path(input_root)
    if not root.exists():
        if verbose:
            print(f"{input_root} does not exist -- not on Kaggle?")
        return []
    specs, seen = [], set()
    for ds in sorted(root.iterdir()):
        if not ds.is_dir():
            continue
        for r in RECIPES:
            try:
                found = r(ds)
            except Exception as e:
                if verbose:
                    print(f"  [{r.__name__}] error in {ds.name}: {str(e)[:90]}")
                continue
            for s in found:
                if s["name"] in seen:
                    continue
                seen.add(s["name"])
                s["kaggle_dir"] = ds.name
                specs.append(s)
                if verbose:
                    print(f"  found '{s['name']}' in {ds.name}")
    return specs
