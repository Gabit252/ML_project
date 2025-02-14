import argparse
import yaml
import pandas as pd
import os 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="params.yaml")
    return parser.parse_args()

def preprocess_data(config):
    raw_path = config['data']['local_path']
    processed_path = "data/processed/processed.csv"
    os.makedirs("data/processed", exist_ok=True)
    df = pd.read_csv(raw_path)
    
    drop_cols = config['preprocess'].get('drop_columns', [])
    df.drop(columns=drop_cols, inplace=True, errors='ignore')
    
    df.columns = df.columns.str.replace(' ', '_')
    df['diagnosis'] = df['diagnosis'].map({'M':0, 'B':1})
    df.to_csv(processed_path, index=False)
    print(f"Preprocessed data saved to {processed_path}")

def main():
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    preprocess_data(config)

if __name__ == "__main__":
    main()