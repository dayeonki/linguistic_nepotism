import json
from pathlib import Path


def compute_comet_averages(directory):
    results = {}
    for file in Path(directory).glob("*.jsonl"):
        total_t, total_c, total_q, total_nc, total_nt = 0.0, 0.0, 0.0, 0.0, 0.0
        count = 0

        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line.strip())
                total_t += sum(data.get("comet_t", []))
                total_c += sum(data.get("comet_c", []))
                total_q += data.get("comet_q", 0.0)
                total_nc += sum(data.get("comet_nc", []))
                total_nt += sum(data.get("comet_nt", []))
                count += 1

        if count > 0:
            avg_t = total_t / (count * len(data.get("comet_t", [1])))
            avg_c = total_c / (count * len(data.get("comet_c", [1])))
            avg_nc = total_nc / (count * len(data.get("comet_nc", [1])))
            avg_nt = total_nt / (count * len(data.get("comet_nt", [1])))
            avg_q = total_q / count
            results[file.name] = {
                "avg_comet_t": avg_t,
                "avg_comet_c": avg_c,
                "avg_comet_q": avg_q,
                "avg_comet_nc": avg_nc,
                "avg_comet_nt": avg_nt,
            }
        else:
            results[file.name] = {
                "avg_comet_t": None,
                "avg_comet_c": None,
                "avg_comet_q": None,
                "avg_comet_nc": None,
                "avg_comet_nt": None,
            }

    return results


if __name__ == "__main__":
    directory_path = "miracl/n1_comet"
    averages = compute_comet_averages(directory_path)

    for filename, stats in averages.items():
        print(f"{filename}:")
        print(f"  Avg comet_q: {stats['avg_comet_q']:.4f}")
        print(f"  Avg comet_t: {stats['avg_comet_t']:.4f}")
        print(f"  Avg comet_c: {stats['avg_comet_c']:.4f}")
        print(f"  Avg comet_nc: {stats['avg_comet_nc']:.4f}")
        print(f"  Avg comet_nt: {stats['avg_comet_nt']:.4f}")
