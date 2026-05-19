import sys
sys.path.insert(0, 'LLMTEXT')
from hf_sku_matcher import HFSKUMatcher, clean_ocr_text
import pandas as pd
from product_matcher import find_top5_matches

print("Loading data...")
df = pd.read_excel('vlm_new2.xlsx')

print("Testing text cleaning...")
cleaned_examples = []
for i in range(min(5, len(df))):
    raw = str(df.iloc[i]['raw_text']) if 'raw_text' in df.columns else str(df.iloc[i]['product_name'])
    cleaned = clean_ocr_text(raw)
    cleaned_examples.append((raw[:80], cleaned))

print(f"Cleaned {len(cleaned_examples)} examples")

print("\nRunning find_top5_matches...")
df_test = df.head(5).copy()
df_test = df_test.rename(columns={'product_name': 'ocr_text'})
df_match = find_top5_matches(df_test, ocr_col='ocr_text')

print(f"Found {len(df_match)} matches")

print("\nInitializing LLM...")
matcher = HFSKUMatcher(batch_size=2)
matcher.load_model()

print("\nProcessing 5 rows...")
result_df = matcher.process_dataframe(df_match, ocr_col='ocr_text', output_col='id_sku_test')

print("\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)
print(f"Total rows: {len(result_df)}")
print(f"Found SKUs: {result_df['id_sku_test'].notna().sum()}")
match_rate = result_df['id_sku_test'].notna().sum() / len(result_df) * 100
print(f"Match rate: {match_rate:.1f}%")

print("\n" + "="*60)
print("DETAILED RESULTS")
print("="*60)

with open('test_results_detailed.txt', 'w', encoding='utf-8') as f:
    f.write("=== OCR CLEANING EXAMPLES ===\n\n")
    for i, (raw, clean) in enumerate(cleaned_examples):
        f.write(f"{i+1}. RAW: {raw}\n")
        f.write(f"   CLEAN: {clean}\n\n")
    
    f.write("\n=== LLM RESULTS ===\n\n")
    for i in range(len(result_df)):
        row = result_df.iloc[i]
        f.write(f"\n{i+1}. OCR: {row['ocr_text'][:60]}\n")
        f.write(f"   Top1: {row.get('top1', 'N/A')}\n")
        f.write(f"   Top1 SKU: {row.get('top1_sku', 'N/A')}\n")
        f.write(f"   LLM SKU: {row['id_sku_test']}\n")
        f.write(f"   Thinking: {str(row.get('llm_thinking', 'N/A'))[:150]}\n")

print("Detailed results saved to test_results_detailed.txt")
print("\nDONE!")
