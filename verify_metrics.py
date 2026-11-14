import os
import time
import librosa
import pandas as pd
from scipy.stats import pearsonr

# Import your Django environment (so you can use your ML models)
import django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
django.setup()

# Import your AI processing pipeline
# (Adjust this import to match your actual PronounceMate service/function)
from Pipeline.pipeline import run_pipeline

def run_verification_test():
    print("Starting PronounceMate System Evaluation...")
    
    # 1. Load the SpeechOcean762 test labels (Human Ground Truth)
    # Assuming you have a CSV with 'filename' and 'human_score' (out of 100)
    dataset_path = "fixtures/speechocean_test_subset.csv"
    audio_dir = "fixtures/speechocean_audio/"
    
    df = pd.read_csv(dataset_path)
    
    ai_scores = []
    human_scores = []
    
    total_audio_duration = 0.0
    total_processing_time = 0.0

    for index, row in df.iterrows():
        filepath = os.path.join(audio_dir, row['filename'])
        script_text = row['transcript']
        human_score = row['human_score']
        
        # Calculate audio duration for RTF
        duration = librosa.get_duration(filename=filepath)
        total_audio_duration += duration
        
        # Start timer
        start_time = time.time()
        
        # Run your AI pipeline (Faster-Whisper + Wav2Vec2 + scoring logic)
        result = run_pipeline(filepath, script_text, skip_denoise=True)
        
        # Stop timer
        end_time = time.time()
        processing_time = end_time - start_time
        total_processing_time += processing_time
        
        
        # Append scores for Pearson correlation
        ai_scores.append(result['score'])
        human_scores.append(human_score)
        
        print(f"Processed {row['filename']} | AI: {result['overall_score']:.1f} | Human: {human_score:.1f} | Time: {processing_time:.2f}s")

    # --- CALCULATE FINAL METRICS ---
    
    # Pearson Correlation
    pcc, _ = pearsonr(human_scores, ai_scores)
    
    # Real-Time Factor (RTF)
    rtf = total_processing_time / total_audio_duration
    
    print("\n" + "="*40)
    print("🏆 FINAL EMPIRICAL VERIFICATION 🏆")
    print("="*40)
    print(f"Total Files Processed: {len(df)}")
    print(f"Total Audio Duration:  {total_audio_duration:.2f} seconds")
    print(f"Total Processing Time: {total_processing_time:.2f} seconds")
    print("-" * 40)
    print(f"✅ Pearson Correlation Coefficient (PCC): {pcc:.3f}")
    print(f"✅ System Real-Time Factor (RTF):       {rtf:.3f}")
    print("="*40)

if __name__ == "__main__":
    run_verification_test()