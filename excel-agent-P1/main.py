from analyzer import analyze_excel
import json

if __name__ == "__main__":
    file_path = "excel-agent-P1/input/Input+metadata.xlsx" 
    summary = analyze_excel(file_path)

    print("🧾 Excel Analysis Summary:")
    print(json.dumps(summary, indent=4))