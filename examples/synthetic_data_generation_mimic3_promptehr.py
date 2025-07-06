"""
end-to-end promptehr pipeline

complete workflow: mimic3 preprocessing -> training -> generation -> evaluation

Usage:
    python promptehr_end_to_end.py --mimic_root /path/to/mimic3 --output_dir ./promptehr_output

This script demonstrates the full pipeline:
1. preprocess mimic3 data
2. train promptehr model  
3. generate synthetic samples
4. evaluate synthetic data quality
5. save results and reports
"""

import argparse
import pickle
import numpy as np
from pathlib import Path
import torch
import warnings
warnings.filterwarnings('ignore')

# import our modules
from promptehr_mimic3_preprocessing import (
    load_mimic3_data, PromptEHRPreprocessingTask, 
    build_code_vocabulary, encode_sequences, split_data, save_preprocessed_data
)
from promptehr_evaluation import evaluate_promptehr_synthetic_data
import sys
sys.path.append(str(Path(__file__).parent.parent))
from pyhealth.models.generators.promptehr import PromptEHRGenerator
from pyhealth.datasets import SampleDataset


def preprocess_mimic3(mimic_root: str, output_dir: str, args) -> str:
    """run mimic3 preprocessing"""
    
    print("="*60)
    print("STEP 1: PREPROCESSING MIMIC-III DATA")
    print("="*60)
    
    # setup paths
    preprocess_dir = Path(output_dir) / "preprocessed"
    preprocess_dir.mkdir(parents=True, exist_ok=True)
    
    # define tables
    tables = ["ADMISSIONS", "DIAGNOSES_ICD"]
    if args.include_procedures:
        tables.append("PROCEDURES_ICD")
    if args.include_medications:
        tables.append("PRESCRIPTIONS")
    
    print(f"loading tables: {tables}")
    
    # load dataset
    dataset = load_mimic3_data(mimic_root, tables)
    
    # preprocessing task
    task = PromptEHRPreprocessingTask(
        max_visits_per_patient=args.max_visits,
        min_visits_per_patient=args.min_visits,
        include_medications=args.include_medications,
        include_procedures=args.include_procedures
    )
    
    # apply preprocessing  
    print("processing patients...")
    sample_dataset = dataset.set_task(task)
    
    if len(sample_dataset.samples) == 0:
        raise ValueError("no samples after preprocessing - check data and parameters")
    
    print(f"processed {len(sample_dataset.samples)} patients")
    
    # build vocabulary
    print("building vocabulary...")
    code_to_idx, idx_to_code = build_code_vocabulary(sample_dataset.samples)
    
    # encode sequences
    print("encoding sequences...")
    encoded_samples = encode_sequences(sample_dataset.samples, code_to_idx)
    
    # split data
    print("splitting data...")
    train_samples, val_samples = split_data(encoded_samples, args.train_ratio)
    
    # save
    save_preprocessed_data(
        str(preprocess_dir),
        train_samples,
        val_samples, 
        code_to_idx,
        idx_to_code
    )
    
    print(f"preprocessing complete! saved to {preprocess_dir}")
    return str(preprocess_dir)


def train_promptehr(preprocess_dir: str, output_dir: str, args) -> str:
    """train promptehr model"""
    
    print("\n" + "="*60)
    print("STEP 2: TRAINING PROMPTEHR MODEL")
    print("="*60)
    
    # setup paths
    model_dir = Path(output_dir) / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # load preprocessed data
    preprocess_path = Path(preprocess_dir)
    
    with open(preprocess_path / "train_samples.pkl", "rb") as f:
        train_samples = pickle.load(f)
    
    with open(preprocess_path / "val_samples.pkl", "rb") as f:
        val_samples = pickle.load(f)
    
    with open(preprocess_path / "metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    
    print(f"loaded {len(train_samples)} train, {len(val_samples)} val samples")
    print(f"vocab size: {metadata['vocab_size']}")
    
    # create sample datasets
    train_dataset = SampleDataset(
        samples=train_samples,
        input_schema={"v": "sequence", "x": "raw"},
        output_schema={}
    )
    
    val_dataset = SampleDataset(
        samples=val_samples,
        input_schema={"v": "sequence", "x": "raw"}, 
        output_schema={}
    )
    
    # init model
    generator = PromptEHRGenerator(
        vocab_size_diag=3000,  # simplified vocab sizes
        vocab_size_proc=1000,
        vocab_size_med=2000,
        n_numerical_features=metadata.get('baseline_feature_dim', 8),
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        max_seq_length=args.max_seq_length
    )
    
    print(f"model config:")
    print(f"  hidden_size: {args.hidden_size}")
    print(f"  num_layers: {args.num_layers}")
    print(f"  max_seq_length: {args.max_seq_length}")
    
    # train
    print("starting training...")
    generator.fit(
        dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.epochs,
        save_dir=str(model_dir)
    )
    
    print(f"training complete! model saved to {model_dir}")
    return str(model_dir)


def generate_synthetic_data(model_dir: str, preprocess_dir: str, output_dir: str, args) -> str:
    """generate synthetic data"""
    
    print("\n" + "="*60)
    print("STEP 3: GENERATING SYNTHETIC DATA")
    print("="*60)
    
    # setup paths
    synthetic_dir = Path(output_dir) / "synthetic"
    synthetic_dir.mkdir(parents=True, exist_ok=True)
    
    # load model
    print(f"loading model from {model_dir}")
    generator = PromptEHRGenerator.load(model_dir)
    
    # load real data for feature distribution
    preprocess_path = Path(preprocess_dir)
    with open(preprocess_path / "train_samples.pkl", "rb") as f:
        train_samples = pickle.load(f)
    
    # extract numerical features from real data
    real_features = []
    for sample in train_samples:
        if "x" in sample and sample["x"]:
            real_features.append(sample["x"])
    
    if real_features:
        real_features = np.array(real_features)
        print(f"using {len(real_features)} real feature vectors")
    else:
        # fallback to random features
        real_features = np.random.randn(args.n_synthetic, 8)
        print("using random features (no real features found)")
    
    # generate
    print(f"generating {args.n_synthetic} synthetic samples...")
    synthetic_samples = generator.generate(
        n_samples=args.n_synthetic,
        max_length=args.max_seq_length,
        temperature=args.temperature,
        numerical_features=real_features
    )
    
    # save synthetic data
    synthetic_path = synthetic_dir / "synthetic_samples.pkl"
    with open(synthetic_path, "wb") as f:
        pickle.dump(synthetic_samples, f)
    
    print(f"generated {len(synthetic_samples)} synthetic samples")
    print(f"saved to {synthetic_path}")
    
    return str(synthetic_path)


def evaluate_synthetic_data(synthetic_path: str, preprocess_dir: str, output_dir: str) -> str:
    """evaluate synthetic data quality"""
    
    print("\n" + "="*60)
    print("STEP 4: EVALUATING SYNTHETIC DATA")
    print("="*60)
    
    # setup paths
    eval_dir = Path(output_dir) / "evaluation" 
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    # load data
    with open(synthetic_path, "rb") as f:
        synthetic_samples = pickle.load(f)
    
    preprocess_path = Path(preprocess_dir)
    with open(preprocess_path / "train_samples.pkl", "rb") as f:
        real_samples = pickle.load(f)
    
    print(f"evaluating {len(synthetic_samples)} synthetic vs {len(real_samples)} real samples")
    
    # run evaluation
    report = evaluate_promptehr_synthetic_data(
        real_samples=real_samples,
        synthetic_samples=synthetic_samples,
        output_dir=str(eval_dir)
    )
    
    print(f"evaluation complete! results saved to {eval_dir}")
    return str(eval_dir)


def create_final_report(output_dir: str, args):
    """create final summary report"""
    
    print("\n" + "="*60)
    print("STEP 5: CREATING FINAL REPORT")
    print("="*60)
    
    output_path = Path(output_dir)
    
    # load evaluation results
    eval_dir = output_path / "evaluation"
    with open(eval_dir / "evaluation_report.pkl", "rb") as f:
        eval_report = pickle.load(f)
    
    # create summary
    summary = {
        "experiment_config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "n_synthetic": args.n_synthetic,
            "temperature": args.temperature,
            "max_visits": args.max_visits,
            "min_visits": args.min_visits,
            "include_procedures": args.include_procedures,
            "include_medications": args.include_medications
        },
        "evaluation_results": eval_report
    }
    
    # save final report
    with open(output_path / "final_report.pkl", "wb") as f:
        pickle.dump(summary, f)
    
    # create readme
    readme_text = f"""
PromptEHR Synthetic EHR Generation Results
==========================================

Experiment Configuration:
- Training epochs: {args.epochs}
- Batch size: {args.batch_size}
- Learning rate: {args.learning_rate}
- Hidden size: {args.hidden_size}
- Number of layers: {args.num_layers}
- Synthetic samples generated: {args.n_synthetic}
- Generation temperature: {args.temperature}

Key Results:
- Fidelity (code correlation): {eval_report['fidelity']['code_freq_correlation']:.4f}
- Utility (TSTR/TRTR ratio): {eval_report['utility']['utility_ratio']:.4f}
- Privacy score: {eval_report['privacy']['privacy_score']:.4f}

Files:
- preprocessed/: Preprocessed MIMIC-III data
- model/: Trained PromptEHR model
- synthetic/: Generated synthetic samples
- evaluation/: Evaluation results and plots
- final_report.pkl: Complete experiment summary

Generated with PyHealth PromptEHR implementation
"""
    
    with open(output_path / "README.txt", "w") as f:
        f.write(readme_text)
    
    print(f"final report saved to {output_path}")
    print("\nExperiment complete! Check README.txt for summary.")


def main():
    parser = argparse.ArgumentParser(description="end-to-end promptehr pipeline")
    
    # data args
    parser.add_argument("--mimic_root", type=str, required=True, help="mimic3 root directory")
    parser.add_argument("--output_dir", type=str, default="./promptehr_output", help="output directory")
    
    # preprocessing args
    parser.add_argument("--include_procedures", action="store_true", help="include procedures")
    parser.add_argument("--include_medications", action="store_true", help="include medications")
    parser.add_argument("--max_visits", type=int, default=10, help="max visits per patient")
    parser.add_argument("--min_visits", type=int, default=2, help="min visits per patient")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="train split ratio")
    
    # model args
    parser.add_argument("--hidden_size", type=int, default=512, help="model hidden size")
    parser.add_argument("--num_layers", type=int, default=4, help="number of transformer layers")
    parser.add_argument("--max_seq_length", type=int, default=256, help="max sequence length")
    
    # training args
    parser.add_argument("--epochs", type=int, default=5, help="training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate")
    
    # generation args
    parser.add_argument("--n_synthetic", type=int, default=1000, help="number of synthetic samples")
    parser.add_argument("--temperature", type=float, default=1.0, help="generation temperature")
    
    # pipeline control
    parser.add_argument("--skip_preprocessing", action="store_true", help="skip preprocessing step")
    parser.add_argument("--skip_training", action="store_true", help="skip training step")
    parser.add_argument("--preprocess_dir", type=str, help="existing preprocessing directory")
    parser.add_argument("--model_dir", type=str, help="existing model directory")
    
    args = parser.parse_args()
    
    # create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("PROMPTEHR END-TO-END PIPELINE")
    print("=============================")
    print(f"output directory: {output_path}")
    print(f"using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    
    # step 1: preprocessing
    if args.skip_preprocessing and args.preprocess_dir:
        preprocess_dir = args.preprocess_dir
        print(f"skipping preprocessing, using {preprocess_dir}")
    else:
        preprocess_dir = preprocess_mimic3(args.mimic_root, args.output_dir, args)
    
    # step 2: training  
    if args.skip_training and args.model_dir:
        model_dir = args.model_dir
        print(f"skipping training, using {model_dir}")
    else:
        model_dir = train_promptehr(preprocess_dir, args.output_dir, args)
    
    # step 3: generation
    synthetic_path = generate_synthetic_data(model_dir, preprocess_dir, args.output_dir, args)
    
    # step 4: evaluation
    eval_dir = evaluate_synthetic_data(synthetic_path, preprocess_dir, args.output_dir)
    
    # step 5: final report
    create_final_report(args.output_dir, args)


if __name__ == "__main__":
    main()