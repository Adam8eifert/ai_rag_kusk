import subprocess
import sys
import os


def run_step(description: str, command: list[str]):
    print(f"\n=== {description} ===")
    print("Spouštím:", " ".join(command))

    result = subprocess.run(command)

    if result.returncode != 0:
        print(f"❌ Chyba ve kroku: {description}")
        sys.exit(1)

    print(f"✅ Hotovo: {description}")


def main():
    print("🚀 Spouštím AI RAG pipeline")

    os.makedirs("data", exist_ok=True)
    os.makedirs("index", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    run_step(
        "Indexování dokumentů",
        [sys.executable, "build_index.py"]
    )

    print("\n🎉 Pipeline dokončena.")
    print("➡️ Pro spuštění API použij:")
    print("   uvicorn app:app --reload")



if __name__ == "__main__":
    main()
