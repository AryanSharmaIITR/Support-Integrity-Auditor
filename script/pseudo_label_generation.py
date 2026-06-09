import pandas as pd
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from dotenv import load_dotenv
import torch.nn.functional as F
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore")

load_dotenv()

print("="*50)
print("Loading Sentence Transformer...")
print("="*50)
encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

model_name = "google/gemma-2-2b-it"

print("="*50)
print("Loading Gemma 2 2B (4-bit quantized)...")
print("="*50)
# 4-bit quantization config — brings VRAM from ~4GB down to ~1.5GB
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="cuda", 
    low_cpu_mem_usage=True,
)

class_labels = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
class_token_ids = [tokenizer.encode(label, add_special_tokens=False)[0] for label in class_labels]

print("✅ Model loaded successfully!\n")


def get_llm_score(ticket_text, ticket_subject):
    # Gemma 2 uses a specific chat template
    # ✅ Correct (Gemma 2 format)
    # Constrained output version - forces single token generation
    prompt = f"""<start_of_turn>system
    You are a severity classifier. Output format: Severity: [TOKEN]

    Available tokens: LOW, MEDIUM, HIGH, CRITICAL

    Classification guidelines:
    - CRITICAL: outage, data loss, security, system down, service unavailable
    - HIGH: broken feature, auth error, payment fails, data mismatch, API 5xx
    - MEDIUM: minor bug, slow response, UI issue, workaround available  
    - LOW: question, feature request, documentation, cosmetic, how-to

    Critical examples: "service outage", "database corrupted", "security breach"
    High examples: "login broken", "checkout failing", "API returns error"
    Medium examples: "button not working", "page slow to load", "filter not applying"
    Low examples: "how to use X?", "can you add Y?", "typo in documentation"

    Classify this ticket:<end_of_turn>
    <start_of_turn>user
    Subject: {ticket_subject}
    Description: {ticket_text}

    Severity: <end_of_turn>
    <start_of_turn>model
    """

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        next_token_logits = outputs.logits[0, -1, :]  # Shape: [vocab_size]
        class_logits = next_token_logits[class_token_ids]  # Shape: [4]
        class_probs = F.softmax(class_logits, dim=-1)

    return {
        "LOW": float(class_probs[0]),
        "MEDIUM": float(class_probs[1]),
        "HIGH": float(class_probs[2]),
        "CRITICAL": float(class_probs[3])
    }

scoring = {
    "LOW": [
        "for new features",
        "is not updating",
        "not receiving the password reset email",
        "How do i"
    ],
    "MEDIUM": [
        "application crashes",
        "data hasn't synced",
        "dashboard is not loading any data",
    ],
    "HIGH": [
        "cannot log into my account even after resetting the password",
        "cannot pass the 2FA check",
        "trying to update payment method",
        "payment processing issue"
    ],
    "CRITICAL": [
        "received a login alert",
        "Lock my account immediately",
        "Someone used my",
        "receiving a 500 Internal Server Error",
    ]
}

def get_embeding_scores(text_desc="Ticket_Description", text_sub="Ticket_Subject"):    
    
    txt = text_sub + " " + text_desc
    encoder_txt = encoder.encode(txt)

    severity_scores = {}
    for key,value in scoring.items():
        score = []
        for sample in value:
            sample_enc = encoder.encode(sample)
            score.append(cosine_similarity([encoder_txt],[sample_enc])[0][0])
        severity_scores[key] = np.max(score) if score else 0

    return severity_scores


Number_category={
    "General Inquiry" : 0,
    "Technical": 1, 
    "Account" : 2, 
    "Billing" : 3,
    "Fraud" : 4
}

def get_cat_resolve_score(cat, resolution_time_hours):
    # Category Score
    cat_num=Number_category[cat]
    if cat_num==0:
        cat_crit,cat_high,cat_med,cat_low = 0.061,0.112,0.144,0.683
    elif cat_num == 1:
        cat_crit,cat_high,cat_med,cat_low = 0.142,0.212,0.665,0.146
    elif cat_num == 2:
        cat_crit,cat_high,cat_med,cat_low = 0.120,0.228,0.149,0.503
    elif cat_num == 3:
        cat_crit,cat_high,cat_med,cat_low = 0.051,0.510,0.224,0.215
    elif cat_num == 4:
        cat_crit,cat_high,cat_med,cat_low = 0.459,0.261,0.023,0.257
    else:
        cat_crit,cat_high,cat_med,cat_low = 0.10,0.25,0.35,0.30
    
    # Resolution Time scores
    if resolution_time_hours < 24:
        time_crit,time_high,time_med,time_low = 0.317,0.278,0.250,0.221
    elif 24 <= resolution_time_hours < 48:
        time_crit,time_high,time_med,time_low = 0.219,0.261,0.256,0.264
    elif 48 <= resolution_time_hours < 72:
        time_crit,time_high,time_med,time_low = 0.266,0,0.336,0.397
    elif resolution_time_hours >= 72:
        time_crit,time_high,time_med,time_low = 0.111,0.357,0.235,0.301
    else:
        time_crit,time_high,time_med,time_low = 0.10,0.25,0.35,0.30
    
    # Combine (average)
    critical = (cat_crit + time_crit)/2
    high = (cat_high + time_high)/2
    medium = (cat_med + time_med)/2
    low = (cat_low + time_low)/2
    
    # Renormalize
    total = critical + high + medium + low
    return {
        "CRITICAL": critical/total,
        "HIGH": high/total,
        "MEDIUM": medium/total,
        "LOW": low/total
    }

def batch_classify_tickets(tickets_df, text_column='Ticket_Description', text_subject='Ticket_Subject',cat_col="Issue_Category", res_col="Resolution_Time_Hours"):
    low_score = []
    med_score = []
    hig_score = []
    crt_score = []

    final_label = []

    for idx, row in tqdm(tickets_df.iterrows(), total=len(tickets_df), desc="Classifying"):
        txt_desc = row[text_column]
        txt_sub = row[text_subject]
        llm_scores = get_llm_score(txt_desc,txt_sub)
        embeding_score = get_embeding_scores(txt_desc,txt_sub)
        cat_res_score = get_cat_resolve_score(row[cat_col],row[res_col])
        
        final_scores = {}
        for label in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
            final_scores[label] = (
                0.3 * llm_scores[label] + 
                0.35 * embeding_score[label] + 
                0.35 * cat_res_score[label]
            )
       
        low_score.append(final_scores["LOW"])
        med_score.append(final_scores["MEDIUM"])
        hig_score.append(final_scores["HIGH"])
        crt_score.append(final_scores["CRITICAL"])

        final_label.append(max(final_scores, key=final_scores.get))

        # Checkpoint every 500 tickets
        if (idx + 1) % 1000 == 0:
            temp_df = tickets_df.iloc[:len(final_label)].copy()
            temp_df['low_score'] = low_score
            temp_df['medium_score'] = med_score
            temp_df["high_score"] = hig_score
            temp_df["critical_score"] = crt_score
            temp_df["Final_severity_label"] = final_label

            temp_df.to_csv("./dataset/checkpoint.csv", index=False)
            print(f"\n💾 Checkpoint saved at {idx + 1} tickets")
    
    tickets_df['low_score'] = low_score
    tickets_df['medium_score'] = med_score
    tickets_df["high_score"] = hig_score
    tickets_df["critical_score"] = crt_score
    tickets_df["Final_severity_label"] = final_label

    return tickets_df

def is_mismatch(row):
    return int(row["Priority_Level"].lower() != row["Final_severity_label"].lower())

def main():
    file_path = "./dataset/enhanced_customer_support_data.csv"
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} tickets")

    # Test on 5 first to measure speed
    import time
    print("\n--- Testing on 5 tickets ---")
    sample = df.head(5)
    start = time.time()
    sample = batch_classify_tickets(sample)
    elapsed = time.time() - start
    print(sample[['Ticket_Subject', 'Final_severity_label',"low_score","medium_score","high_score","critical_score"]].to_string())
    print(f"\nTime per ticket: {elapsed/5:.1f}s")
    print(f"Estimated total: {elapsed/5 * len(df) / 3600:.1f} hours")

    proceed = input("\nProceed with full dataset? (y/n): ")
    if proceed.lower() == 'y':
        # df = batch_classify_tickets(df)
        df["Is_Mismatch"] = df.apply(is_mismatch, axis=1)
        df.to_csv("./dataset/preprocessed_data.csv", index=False)
        print("\nFinal_severity_label Distribution:")
        print(df['Final_severity_label'].value_counts().sort_index())
        print("\Is_Mismatch Distribution:")
        print(df['Is_Mismatch'].value_counts().sort_index())

if __name__=="__main__":
    main()