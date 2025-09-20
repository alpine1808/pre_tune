import argparse
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="pre_tune", description="Tiền xử lý làm sạch chunk (Gemini-dominant).")
    p.add_argument("--filename", required=True, help="Tên file để tra trong documents.data->>'filename'")
    p.add_argument("--json-out", default="-", help="Đường dẫn JSON output; '-' = stdout")
    return p.parse_args()
