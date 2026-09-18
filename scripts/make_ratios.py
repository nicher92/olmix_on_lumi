import json
import csv
import os

def main():
    json_path = 'data/lumi_nested_variants.json'
    output_csv = 'ratios_big.csv'  # Naming it exactly what your fit_config expects

    if not os.path.exists(json_path):
        print(f"Error: Could not find {json_path}")
        return

    # 1. Load the mapped JSON data
    with open(json_path, 'r') as f:
        variants = json.load(f)

    # 2. Extract the clean dataset names from the first variant
    clean_domains = sorted(list(variants[0]['mix'].keys()))
    headers = ['run'] + clean_domains

    # 3. Write out the clean CSV
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        
        for variant in variants:
            row = {'run': variant['variant_id']}
            for domain in clean_domains:
                # Extract just the weight float from the nested dict
                row[domain] = variant['mix'][domain]['weight']
            writer.writerow(row)

    print(f"✅ Success! Mapped {len(variants)} runs across {len(clean_domains)} clean datasets.")
    print(f"✅ Saved to {output_csv}")

if __name__ == "__main__":
    main()
