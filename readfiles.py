from pathlib import Path

out_name = "files.txt"


def scan(folder, level=0, lines=None):
    if lines is None:
        lines = []

    # 按名称排序，文件夹排在文件前面
    items = sorted(folder.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))

    for item in items:
        # 避免把输出文件本身也写进去
        if item.name == out_name:
            continue

        space = "    " * level

        if item.is_dir():
            lines.append(f"{space}{item.name}/")
            scan(item, level + 1, lines)
        else:
            lines.append(f"{space}{item.name}")

    return lines


def main():
    now = Path(".")
    lines = [f"{now.resolve().name}/"]
    lines += scan(now, 1)

    with open(out_name, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"已生成目录文件：{out_name}")


if __name__ == "__main__":
    main()