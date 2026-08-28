from datasets import load_dataset

def fetch_sample_vignettes():
    print("Downloading MedQA dataset(this might take a minute)...")
    dataset=load_dataset("GBaker/MedQA-USMLE-4-options", split="train")

    sample_cases = dataset.shuffle(seed=42).select(range(5))

    print("\nHere are your 5 raw clinical notes for Stage 1:")
    for i,case in enumerate(sample_cases):
        print(f"\n--- Case {i+1} ---")
        print(case['question'])
if __name__ == "__main__":
    fetch_sample_vignettes()