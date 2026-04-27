"""
Modrinth Mod JAR 批量下载器
用途：下载 Minecraft mod 的 JAR 文件，用于提取贴图素材

依赖安装：
    pip install httpx tqdm

用法：
    python modrinth_downloader.py                    # 默认下载前500个热门mod
    python modrinth_downloader.py --limit 200        # 只下载200个
    python modrinth_downloader.py --loader fabric    # 只下载 Fabric mod
    python modrinth_downloader.py --out ./my_mods    # 指定输出目录
"""

import httpx
import asyncio
import argparse
import json
import time
from pathlib import Path
from tqdm import tqdm


# ── 配置 ──────────────────────────────────────────────────────────────────────

BASE_URL    = "https://api.modrinth.com/v2"
USER_AGENT  = "mc-texture-dataset-builder/1.0 (training research)"  # Modrinth 要求标注 UA
CONCURRENCY = 8    # 同时下载的并发数，不要设太高（速率限制300次/分钟）
RETRY_MAX   = 3    # 失败重试次数
RETRY_DELAY = 2.0  # 重试等待秒数


# ── 搜索 mod 列表 ──────────────────────────────────────────────────────────────

async def search_mods(client: httpx.AsyncClient, loader: str, limit: int) -> list[dict]:
    """
    分页搜索 Modrinth，返回 mod 基本信息列表
    按下载量排序，优先获取素材最丰富的热门 mod
    """
    mods = []
    page_size = 100  # API 单页最大值
    offset = 0

    # facets 过滤条件：只要 mod 类型，指定 loader
    facets = json.dumps([
        ["project_type:mod"],
        [f"categories:{loader}"]
    ])

    print(f"🔍 搜索 {loader} mod，目标数量：{limit}")

    with tqdm(total=limit, desc="搜索进度") as pbar:
        while len(mods) < limit:
            fetch_count = min(page_size, limit - len(mods))
            params = {
                "facets": facets,
                "limit":  fetch_count,
                "offset": offset,
                "index":  "downloads",  # 按下载量排序
            }

            resp = await client.get(f"{BASE_URL}/search", params=params)
            resp.raise_for_status()
            data = resp.json()

            hits = data.get("hits", [])
            if not hits:
                break  # 没有更多结果

            mods.extend(hits)
            pbar.update(len(hits))
            offset += len(hits)

            # 礼貌性延迟，避免触发速率限制
            await asyncio.sleep(0.3)

    print(f"✅ 共找到 {len(mods)} 个 mod")
    return mods


# ── 获取下载链接 ───────────────────────────────────────────────────────────────

async def get_jar_url(client: httpx.AsyncClient, project_id: str, loader: str) -> tuple[str, str] | None:
    """
    给定 project_id，返回 (文件名, 下载URL)
    优先选最新版本，选主文件（非 source jar）
    """
    for attempt in range(RETRY_MAX):
        try:
            resp = await client.get(
                f"{BASE_URL}/project/{project_id}/version",
                params={"loaders": json.dumps([loader])}
            )
            resp.raise_for_status()
            versions = resp.json()

            if not versions:
                return None

            # 取最新版本的第一个文件（通常是主 JAR）
            latest = versions[0]
            for file in latest.get("files", []):
                name = file["filename"]
                # 跳过 sources jar（没有贴图）
                if "sources" in name or "javadoc" in name:
                    continue
                return name, file["url"]

            return None

        except (httpx.HTTPError, KeyError):
            if attempt < RETRY_MAX - 1:
                await asyncio.sleep(RETRY_DELAY)

    return None


# ── 下载单个 JAR ───────────────────────────────────────────────────────────────

async def download_jar(
    client:   httpx.AsyncClient,
    filename: str,
    url:      str,
    out_dir:  Path,
    sem:      asyncio.Semaphore,
) -> bool:
    """下载 JAR 到本地，已存在则跳过"""
    dest = out_dir / filename

    # 已下载则跳过（支持断点续跑）
    if dest.exists():
        return True

    async with sem:
        for attempt in range(RETRY_MAX):
            try:
                async with client.stream("GET", url) as resp:
                    resp.raise_for_status()
                    with open(dest, "wb") as f:
                        async for chunk in resp.aiter_bytes(chunk_size=65536):
                            f.write(chunk)
                return True

            except (httpx.HTTPError, OSError):
                if dest.exists():
                    dest.unlink()  # 清理残缺文件
                if attempt < RETRY_MAX - 1:
                    await asyncio.sleep(RETRY_DELAY)

    return False


# ── 主流程 ─────────────────────────────────────────────────────────────────────

async def main(loader: str, limit: int, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "download_log.jsonl"  # 记录每个 mod 的下载结果

    headers = {"User-Agent": USER_AGENT}

    # 超时设置：连接5秒，读取120秒（大文件下载）
    timeout = httpx.Timeout(connect=5.0, read=120.0, write=10.0, pool=5.0)

    async with httpx.AsyncClient(headers=headers, timeout=timeout, follow_redirects=True) as client:

        # 第一步：搜索所有 mod
        mods = await search_mods(client, loader, limit)

        # 第二步：并发获取下载链接
        print("\n🔗 获取下载链接...")
        sem = asyncio.Semaphore(CONCURRENCY)

        async def fetch_url(mod):
            pid   = mod["project_id"]
            slug  = mod["slug"]
            title = mod["title"]
            result = await get_jar_url(client, pid, loader)
            return pid, slug, title, result

        url_tasks = [fetch_url(mod) for mod in mods]
        url_results = []
        for coro in tqdm(asyncio.as_completed(url_tasks), total=len(url_tasks), desc="获取链接"):
            url_results.append(await coro)

        # 过滤掉没有找到链接的
        valid = [(pid, slug, title, fname, url)
                 for pid, slug, title, res in url_results
                 if res is not None
                 for fname, url in [res]]

        print(f"✅ 有效下载链接：{len(valid)} / {len(mods)}")

        # 第三步：并发下载所有 JAR
        print("\n⬇️  开始下载 JAR 文件...")

        async def dl(item):
            pid, slug, title, fname, url = item
            ok = await download_jar(client, fname, url, out_dir, sem)
            # 记录日志
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "project_id": pid,
                    "slug":       slug,
                    "title":      title,
                    "filename":   fname,
                    "url":        url,
                    "success":    ok,
                }, ensure_ascii=False) + "\n")
            return ok

        dl_tasks = [dl(item) for item in valid]
        success = 0
        with tqdm(total=len(dl_tasks), desc="下载进度", unit="jar") as pbar:
            for coro in asyncio.as_completed(dl_tasks):
                ok = await coro
                if ok:
                    success += 1
                pbar.update(1)
                pbar.set_postfix({"成功": success})

    print(f"\n🎉 完成！成功下载 {success} / {len(valid)} 个 JAR")
    print(f"📁 文件位置：{out_dir.resolve()}")
    print(f"📝 下载日志：{log_path.resolve()}")


# ── 入口 ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Modrinth Mod JAR 批量下载器")
    parser.add_argument("--loader", default="fabric",
                        choices=["fabric", "forge", "neoforge", "quilt"],
                        help="Mod 加载器类型（默认 fabric）")
    parser.add_argument("--limit",  type=int, default=500,
                        help="最多下载多少个 mod（默认 500）")
    parser.add_argument("--out",    default="./jars",
                        help="JAR 文件保存目录（默认 ./jars）")
    args = parser.parse_args()

    asyncio.run(main(
        loader  = args.loader,
        limit   = args.limit,
        out_dir = Path(args.out),
    ))
