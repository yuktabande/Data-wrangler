# main.py
import json
import os
import sys
from pathlib import Path
import pandas as pd
from agent.loader import summarize_excel, sample_sheets
from agent.suggestions import generate_suggestions
from agent.executor import TaskExecutor, execute_natural_language
from utils.file_handler import ensure_output_dir, resolve_input_path, write_text_file
from utils.formatter import pretty_title

def display_menu(suggestions_text: str) -> None:
    """Display interactive menu for Phase 3"""
    print(pretty_title("Phase 3 → Task Execution"))
    print("🚀 Ready to execute tasks! Choose an option:\n")
    
    # Parse suggestions into numbered list
    suggestions_lines = [line.strip() for line in suggestions_text.split('\n') if line.strip()]
    numbered_suggestions = []
    
    for i, line in enumerate(suggestions_lines, 1):
        if line and not line.startswith('⚠️'):
            numbered_suggestions.append(f"{i}. {line}")
    
    print("📋 AI Suggested Actions:")
    for suggestion in numbered_suggestions[:8]:  # Show up to 8 suggestions
        print(f"   {suggestion}")
    
    print(f"\n📝 Custom Options:")
    print(f"   {len(numbered_suggestions) + 1}. Enter custom instruction")
    print(f"   {len(numbered_suggestions) + 2}. Execute multiple tasks")
    print(f"   0. Exit")
    
    return numbered_suggestions

def get_user_choice(max_options: int) -> int:
    """Get and validate user choice"""
    while True:
        try:
            choice = input(f"\n🎯 Choose an option (0-{max_options}): ").strip()
            choice_num = int(choice)
            if 0 <= choice_num <= max_options:
                return choice_num
            else:
                print(f"❌ Please enter a number between 0 and {max_options}")
        except ValueError:
            print("❌ Please enter a valid number")

def execute_selected_task(executor: TaskExecutor, instruction: str) -> None:
    """Execute a single task and display results"""
    print(f"\n⚙️ Executing: {instruction}")
    print("━" * 50)
    
    try:
        result = execute_natural_language(executor.file_path, instruction)
        
        # Display parsing results
        print("🧠 Parsed Instruction:")
        parsed = result["parsed_instruction"]
        print(f"   Action: {parsed.get('action', 'unknown')}")
        print(f"   Sheets: {parsed.get('sheets', [])}")
        print(f"   Columns: {parsed.get('columns', [])}")
        
        # Display execution results
        print("\n📊 Execution Results:")
        exec_result = result["execution_result"]
        
        if exec_result["status"] == "success":
            print(f"   ✅ {exec_result['message']}")
            
            # Show additional details based on action type
            if "output_path" in exec_result:
                print(f"   📁 Output saved: {exec_result['output_path']}")
            
            if "result_shape" in exec_result:
                print(f"   📏 Result shape: {exec_result['result_shape']}")
            
            if "cleaning_steps" in exec_result:
                print("   🧹 Cleaning steps performed:")
                for step in exec_result["cleaning_steps"]:
                    print(f"      • {step}")
            
            if "chart_types" in exec_result:
                print(f"   📈 Charts created: {', '.join(exec_result['chart_types'])}")
                
        else:
            print(f"   ❌ {exec_result['message']}")
    
    except Exception as e:
        print(f"   ❌ Execution failed: {str(e)}")

def batch_execution_mode(executor: TaskExecutor, suggestions: list) -> None:
    """Handle multiple task execution"""
    print("\n🔄 Batch Execution Mode")
    print("Enter task numbers separated by commas (e.g., 1,3,5)")
    print("Or enter 'all' to execute all suggestions")
    
    batch_input = input("Tasks to execute: ").strip().lower()
    
    tasks_to_execute = []
    
    if batch_input == "all":
        tasks_to_execute = suggestions
    else:
        try:
            task_numbers = [int(x.strip()) for x in batch_input.split(',')]
            for num in task_numbers:
                if 1 <= num <= len(suggestions):
                    # Extract the actual suggestion text (remove numbering)
                    suggestion_text = suggestions[num-1].split('. ', 1)[1] if '. ' in suggestions[num-1] else suggestions[num-1]
                    tasks_to_execute.append(suggestion_text)
                else:
                    print(f"⚠️ Skipping invalid task number: {num}")
        except ValueError:
            print("❌ Invalid format. Use comma-separated numbers or 'all'")
            return
    
    if not tasks_to_execute:
        print("❌ No valid tasks selected")
        return
    
    print(f"\n🚀 Executing {len(tasks_to_execute)} tasks...")
    
    results_summary = []
    
    for i, task in enumerate(tasks_to_execute, 1):
        print(f"\n{'='*60}")
        print(f"Task {i}/{len(tasks_to_execute)}")
        execute_selected_task(executor, task)
        
        # Brief pause for readability
        if i < len(tasks_to_execute):
            input("\nPress Enter to continue to next task...")
    
    print(f"\n✅ Batch execution completed! {len(tasks_to_execute)} tasks processed.")

def save_execution_log(results: list) -> None:
    """Save execution history to file"""
    log_data = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "executed_tasks": results
    }
    
    log_path = Path("output") / "execution_log.json"
    with open(log_path, 'w') as f:
        json.dump(log_data, f, indent=2, default=str)
    
    print(f"📝 Execution log saved: {log_path}")

def main():
    # --- Phase 1: Analyze ---
    file_path = "excel-agent/data/input.xlsx"
    
    if not os.path.exists(file_path):
        print("❌ Input file not found. Please check the path:")
        print(f"   Expected: {file_path}")
        return

    print(pretty_title("Phase 1 → Analyze"))
    summary = summarize_excel(file_path)
    ensure_output_dir()
    
    summary_path = os.path.join("output", "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4, default=str)
    print(f"📝 Saved summary → {summary_path}")

    # Get small samples for LLM context
    samples = sample_sheets(file_path, n=5)

    # --- Phase 2: AI Suggestions ---
    print(pretty_title("Phase 2 → AI Suggestions"))
    try:
        suggestions_text = generate_suggestions(summary, samples)
    except Exception as e:
        suggestions_text = f"⚠️ Could not fetch AI suggestions: {e}"
        print(suggestions_text)
        return

    print("\n🔍 AI Suggested Automations:\n")
    print(suggestions_text)

    # Save suggestions
    suggest_path = os.path.join("output", "suggestions.txt")
    write_text_file(suggest_path, suggestions_text)
    print(f"\n💾 Saved suggestions → {suggest_path}")

    # --- Phase 3: Task Execution ---
    try:
        # Initialize executor
        executor = TaskExecutor(file_path)
        
        # Display menu and handle user interaction
        numbered_suggestions = display_menu(suggestions_text)
        max_options = len(numbered_suggestions) + 2
        
        while True:
            choice = get_user_choice(max_options)
            
            if choice == 0:
                print("\n👋 Goodbye!")
                break
            elif 1 <= choice <= len(numbered_suggestions):
                # Execute selected suggestion
                suggestion_text = numbered_suggestions[choice-1].split('. ', 1)[1]
                execute_selected_task(executor, suggestion_text)
                
                # Ask if user wants to continue
                continue_choice = input("\n❓ Execute another task? (y/n): ").strip().lower()
                if continue_choice != 'y':
                    break
                    
            elif choice == len(numbered_suggestions) + 1:
                # Custom instruction
                print("\n💭 Enter your custom instruction:")
                custom_instruction = input("Instruction: ").strip()
                if custom_instruction:
                    execute_selected_task(executor, custom_instruction)
                    
                    # Save custom instruction
                    write_text_file(
                        os.path.join("output", "custom_instruction.txt"), 
                        custom_instruction
                    )
                else:
                    print("❌ No instruction provided")
                
                # Ask if user wants to continue
                continue_choice = input("\n❓ Execute another task? (y/n): ").strip().lower()
                if continue_choice != 'y':
                    break
                    
            elif choice == len(numbered_suggestions) + 2:
                # Batch execution mode
                batch_execution_mode(executor, numbered_suggestions)
                break
            
    except Exception as e:
        print(f"❌ Error initializing executor: {str(e)}")
        print("Make sure your input file exists and is accessible.")

if __name__ == "__main__":
    main()