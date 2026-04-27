"""
Minecraft Mod JAR 贴图提取器
从批量下载的 JAR 文件中提取贴图，清洗并整理为训练数据集

依赖安装：
    pip install Pillow tqdm

用法：
    python extract_textures.py                        # 默认从 ./jars 提取到 ./dataset
    python extract_textures.py --jars ./my_jars       # 指定 JAR 目录
    python extract_textures.py --size 16              # 只保留 16×16 图片
    python extract_textures.py --size 16 32           # 保留 16×16 和 32×32
    python extract_textures.py --category item block  # 只提取物品和方块贴图

输出结构：
    dataset/
    ├── images/
    │   ├── minecraft__item__apple.png
    │   ├── create__item__wrench.png
    │   └── ...
    └── labels.jsonl     ← 每张图对应的元数据（mod名、分类、文件名、尺寸等）
"""

import zipfile
import argparse
import json
import re
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from io import BytesIO


# ── 配置 ──────────────────────────────────────────────────────────────────────

# 要提取的贴图子目录（相对于 assets/<mod_id>/textures/）
CATEGORIES = ["item", "block", "fluid"]

# 动画贴图特征：高度是宽度的整数倍（且 > 1），说明是帧序列，跳过
ANIM_RATIO_MAX = 2  # 高度/宽度 超过此值认为是动画帧，丢弃

# 输出图片统一转为 RGBA（保留透明通道）
OUTPUT_MODE = "RGBA"


# ── 单个 JAR 处理 ──────────────────────────────────────────────────────────────

def extract_from_jar(
    jar_path:    Path,
    out_dir:     Path,
    target_sizes: set[int],
    categories:  list[str],
    log_records: list[dict],
):
    """
    从一个 JAR 文件中提取符合条件的贴图
    返回（成功数, 跳过数）
    """
    success = 0
    skipped = 0

    try:
        with zipfile.ZipFile(jar_path, "r") as zf:
            all_files = zf.namelist()

            # 筛选出目标路径的 PNG 文件
            # 路径格式：assets/<mod_id>/textures/<category>/xxx.png
            targets = [
                f for f in all_files
                if f.startswith("assets/")
                and "/textures/" in f
                and f.endswith(".png")
                and any(f"/textures/{cat}/" in f for cat in categories)
            ]

            for fpath in targets:
                # 解析路径信息
                # assets / <mod_id> / textures / <category> / [sub/] <name>.png
                parts = fpath.split("/")
                try:
                    assets_idx   = parts.index("assets")
                    textures_idx = parts.index("textures")
                    mod_id   = parts[assets_idx + 1]
                    category = parts[textures_idx + 1]
                    # 子目录中的文件，用 __ 连接（如 armor/chainmail_layer_1）
                    name_parts = parts[textures_idx + 2:]
                    raw_name   = "__".join(name_parts).removesuffix(".png")
                except (ValueError, IndexError):
                    skipped += 1
                    continue

                # 读取图片
                try:
                    data = zf.read(fpath)
                    img  = Image.open(BytesIO(data))
                except Exception:
                    skipped += 1
                    continue

                w, h = img.size

                # ── 过滤规则 ──────────────────────────────────

                # 1. 必须是正方形或接近正方形（排除长条 GUI 元素）
                #    对 block/item 贴图放宽：允许高度是宽度的整数倍（动画帧）
                if w == 0 or h == 0:
                    skipped += 1
                    continue

                ratio = h / w
                # 动画帧贴图：高度是宽度整数倍，且倍数 > ANIM_RATIO_MAX
                if ratio > ANIM_RATIO_MAX and ratio == int(ratio):
                    skipped += 1
                    continue

                # 2. 尺寸过滤（如果指定了目标尺寸）
                if target_sizes and w not in target_sizes:
                    skipped += 1
                    continue

                # 3. 排除明显的占位图（全透明 或 单色且极小）
                img_rgba = img.convert("RGBA")
                pixels   = img_rgba.getdata()
                alphas   = [p[3] for p in pixels]
                if max(alphas) == 0:
                    # 全透明，没有意义
                    skipped += 1
                    continue

                # ── 保存 ──────────────────────────────────────

                # 文件名：mod_id__category__texture_name.png
                # 用 __ 分隔，之后建标签时好解析
                safe_mod = re.sub(r"[^\w\-]", "_", mod_id)
                out_name = f"{safe_mod}__{category}__{raw_name}.png"
                out_path = out_dir / out_name

                # 已存在跳过（同名贴图可能来自多个版本的同一 mod）
                if out_path.exists():
                    skipped += 1
                    continue

                img_rgba.save(out_path, "PNG")

                # 记录元数据
                log_records.append({
                    "filename":  out_name,
                    "mod_id":    mod_id,
                    "category":  category,
                    "raw_name":  raw_name,
                    "width":     w,
                    "height":    h,
                    "jar":       jar_path.name,
                    # 简单标签：下划线换空格（后续可用 lang 文件替换为真实名称）
                    "label":     raw_name.replace("__", " ").replace("_", " "),
                })

                success += 1

    except zipfile.BadZipFile:
        print(f"  ⚠️  损坏的 ZIP/JAR，跳过：{jar_path.name}")

    return success, skipped


# ── 语言文件提取（可选，用于更好的标签）─────────────────────────────────────────

def extract_lang(jar_path: Path) -> dict[str, str]:
    """
    提取 JAR 中的 en_us.json 语言文件
    返回 { "item.modid.item_name": "显示名称", ... }
    """
    lang = {}
    try:
        with zipfile.ZipFile(jar_path, "r") as zf:
            lang_files = [f for f in zf.namelist()
                          if f.endswith("lang/en_us.json")]
            for lf in lang_files:
                data = json.loads(zf.read(lf).decode("utf-8", errors="ignore"))
                lang.update(data)
    except Exception:
        pass
    return lang


def apply_lang_labels(records: list[dict], lang: dict[str, str]):
    """
    用语言文件中的真实名称替换文件名生成的简单标签
    key 格式：item.modid.item_name 或 block.modid.block_name
    """
    for rec in records:
        mod_id   = rec["mod_id"]
        category = rec["category"]  # item / block
        # raw_name 可能有子目录前缀，取最后一段作为 id
        item_id  = rec["raw_name"].split("__")[-1]
        key      = f"{category}.{mod_id}.{item_id}"
        if key in lang:
            rec["label"] = lang[key]


# ── 主流程 ─────────────────────────────────────────────────────────────────────

def main(jar_dir: Path, out_dir: Path, target_sizes: list[int], categories: list[str]):
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_path = out_dir / "labels.jsonl"

    jar_files = sorted(jar_dir.glob("*.jar"))
    if not jar_files:
        print(f"❌ 在 {jar_dir} 中没有找到任何 .jar 文件")
        return

    print(f"📦 找到 {len(jar_files)} 个 JAR 文件")
    print(f"🎯 目标分类：{categories}")
    print(f"📐 目标尺寸：{target_sizes if target_sizes else '全部'}")
    print()

    size_set     = set(target_sizes)
    all_records  = []
    total_ok     = 0
    total_skip   = 0

    for jar_path in tqdm(jar_files, desc="处理 JAR", unit="jar"):
        records = []
        ok, skip = extract_from_jar(jar_path, images_dir, size_set, categories, records)

        # 尝试用语言文件优化标签
        if records:
            lang = extract_lang(jar_path)
            if lang:
                apply_lang_labels(records, lang)

        all_records.extend(records)
        total_ok   += ok
        total_skip += skip

    # 写入标签文件
    with open(labels_path, "w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # 统计尺寸分布
    from collections import Counter
    size_dist = Counter(f"{r['width']}x{r['height']}" for r in all_records)

    print(f"\n🎉 提取完成！")
    print(f"   ✅ 保存图片：{total_ok}")
    print(f"   ⏭️  跳过图片：{total_skip}")
    print(f"   📐 尺寸分布：{dict(size_dist.most_common(8))}")
    print(f"   📁 图片目录：{images_dir.resolve()}")
    print(f"   📝 标签文件：{labels_path.resolve()}")


# ── 入口 ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Minecraft Mod JAR 贴图提取器")
    parser.add_argument("--jars",     default="./jars",
                        help="JAR 文件目录（默认 ./jars）")
    parser.add_argument("--out",      default="./dataset",
                        help="输出目录（默认 ./dataset）")
    parser.add_argument("--size",     type=int, nargs="*",
                        help="只保留指定尺寸，如 --size 16 32（不填则保留全部）")
    parser.add_argument("--category", nargs="*", default=CATEGORIES,
                        choices=["item", "block", "fluid", "entity", "gui"],
                        help=f"提取的贴图分类（默认 {CATEGORIES}）")
    args = parser.parse_args()

    main(
        jar_dir      = Path(args.jars),
        out_dir      = Path(args.out),
        target_sizes = args.size or [],
        categories   = args.category,
    )
