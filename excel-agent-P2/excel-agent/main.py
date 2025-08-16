import json
import os
import sys

from agent.loader import summarize_excel, sample_sheets
from agent.suggestions import generate_suggestions
from utils.file_handler import ensure_output_dir, resolve_input_path, write_text_file
from utils.formatter import pretty_title

def main():
    # --- CLI arg or default ---
    file_path = "excel-agent/data/Input+metadata.xlsx"

    print(pretty_title("Phase 1 → Analyze"))
    summary = summarize_excel(file_path)
    ensure_output_dir()
    summary_path = os.path.join("output", "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"📝 saved summary → {summary_path}")

    # (optional) small samples (first 5 rows per sheet)
    samples = sample_sheets(file_path, n=5)

    # --- Phase 2: LLM suggestions ---
    print(pretty_title("Phase 2 → Gemini Suggestions"))
    try:
        suggestions_text = generate_suggestions(summary, samples)
    except Exception as e:
        suggestions_text = f"⚠️ could not fetch LLM suggestions: {e}"
    print("\n🔍 Suggested automations:\n")
    print(suggestions_text)

    # save suggestions to file
    suggest_path = os.path.join("output", "suggestions.txt")
    write_text_file(suggest_path, suggestions_text)
    print(f"\n💾 saved suggestions → {suggest_path}")

    # --- custom input capture ---
    print()
    answer = input("Do you want to add any custom suggestions? (yes/no): ").strip().lower()
    if answer == "yes":
        user_suggestion = input("Please enter your suggestion: ").strip()
        print("\n✅ Custom suggestion recorded:")
        print(user_suggestion)
        write_text_file(os.path.join("output", "custom_suggestion.txt"), user_suggestion)
    else:
        print("\n👍 No custom suggestions added.")

if __name__ == "__main__":
    main()