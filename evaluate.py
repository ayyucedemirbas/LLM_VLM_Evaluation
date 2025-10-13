import pandas as pd
import numpy as np
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import nltk
import warnings

warnings.filterwarnings('ignore')

try:
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    nltk.download('punkt', quiet=True)
except:
    pass

def calculate_corpus_bleu(references, candidates):
    ref_tokens = [[ref.lower().split()] for ref in references]
    cand_tokens = [cand.lower().split() for cand in candidates]
    
    smoothing = SmoothingFunction().method1
    
    bleu1 = corpus_bleu(ref_tokens, cand_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothing)
    bleu2 = corpus_bleu(ref_tokens, cand_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
    bleu3 = corpus_bleu(ref_tokens, cand_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing)
    bleu4 = corpus_bleu(ref_tokens, cand_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)
    
    return bleu1, bleu2, bleu3, bleu4

def calculate_corpus_meteor(references, candidates):
    meteor_scores = []
    for ref, cand in zip(references, candidates):
        try:
            score = meteor_score([ref.lower().split()], cand.lower().split())
            meteor_scores.append(score)
        except:
            meteor_scores.append(0.0)
    
    return np.mean(meteor_scores)

def calculate_corpus_rouge(references, candidates):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    
    for ref, cand in zip(references, candidates):
        scores = scorer.score(ref, cand)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)
    
    return np.mean(rouge1_scores), np.mean(rouge2_scores), np.mean(rougeL_scores)

def calculate_corpus_bertscore(references, candidates, batch_size=64):
    
    import torch
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Use a lighter model for faster computation
    # Options: 'roberta-large' (accurate but slow), 'bert-base-uncased' (fast), 'distilbert-base-uncased' (fastest)
    model_type = 'bert-base-uncased'
    
    P, R, F1 = bert_score(
        candidates, 
        references, 
        lang='en', 
        model_type=model_type,
        num_layers=5,  # Use fewer layers for speed (default is all layers)
        verbose=False, 
        device=device,
        batch_size=batch_size,
        nthreads=4,  # Use multiple threads
        rescale_with_baseline=False  # Skip baseline rescaling for speed
    )
    
    return P.mean().item(), R.mean().item(), F1.mean().item()

def evaluate_vlm_results(input_file, output_file='evaluation_results.txt'):
    df = pd.read_csv(input_file)
    
    if 'ground_truth' not in df.columns or 'prediction' not in df.columns:
        raise ValueError("CSV must contain 'ground_truth' and 'prediction' columns")
    
    df['ground_truth'] = df['ground_truth'].fillna('')
    df['prediction'] = df['prediction'].fillna('')
    
    df['ground_truth'] = df['ground_truth'].astype(str)
    df['prediction'] = df['prediction'].astype(str)
    
    references = df['ground_truth'].tolist()
    candidates = df['prediction'].tolist()
    

    bleu1, bleu2, bleu3, bleu4 = calculate_corpus_bleu(references, candidates)
    
    meteor = calculate_corpus_meteor(references, candidates)
    
    rouge1, rouge2, rougeL = calculate_corpus_rouge(references, candidates)
    
    bert_p, bert_r, bert_f1 = calculate_corpus_bertscore(references, candidates)
    
    results = {
        'BLEU-1': bleu1,
        'BLEU-2': bleu2,
        'BLEU-3': bleu3,
        'BLEU-4': bleu4,
        'METEOR': meteor,
        'ROUGE-1': rouge1,
        'ROUGE-2': rouge2,
        'ROUGE-L': rougeL,
        'BERTScore_Precision': bert_p,
        'BERTScore_Recall': bert_r,
        'BERTScore_F1': bert_f1
    }
    
    # Print results
    print("\n" + "="*70)
    print("EVALUATION RESULTS - CORPUS-LEVEL METRICS")
    print("="*70)
    print(f"Total samples: {len(df)}\n")
    
    for metric, value in results.items():
        print(f"{metric:25s}: {value:.6f}")
    
    print("="*70)
    
    with open(output_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("EVALUATION RESULTS - CORPUS-LEVEL METRICS\n")
        f.write("="*70 + "\n")
        f.write(f"Input file: {input_file}\n")
        f.write(f"Total samples: {len(df)}\n\n")
        
        for metric, value in results.items():
            f.write(f"{metric:25s}: {value:.6f}\n")
        
        f.write("="*70 + "\n")
    
    print(f"\nResults saved to: {output_file}")
    
    return results

if __name__ == "__main__":
    input_csv = "test_predictions_novel.csv"
    output_txt = "vlm_evaluation_results.txt"
    
    results = evaluate_vlm_results(input_csv, output_txt)
