import os
import sys
import pandas as pd
import logging
from pathlib import Path
from datasets import Dataset

# --- PATH PRE-REQUISITE ---
# scripts/evaluate_ragas.py is located inside the 'scripts' directory.
# Resolve the project root to allow absolute imports from the 'src' package.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Local Imports
try:
    from src.rag.agent import build_agent
    from langchain_ollama import ChatOllama, OllamaEmbeddings
    from ragas import evaluate
    # 
    from ragas.metrics import faithfulness, answer_relevancy
    # raga >= 0.2.x uses LangchainLLMWrapper and LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
except ImportError as e:
    print(f"Error: Missing dependencies. Please run: pip install ragas datasets")
    print(f"Details: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==========================================
# 1. LOCAL MODEL WRAPPERS
# ==========================================
logger.info("Configuring local Ollama models for RAGAS evaluation...")

# Critic/Judge LLM
llm = ChatOllama(model="llama3.1", temperature=0)
critic_llm = LangchainLLMWrapper(llm)

# Evaluator Embeddings
embeddings = OllamaEmbeddings(model="mxbai-embed-large")
critic_embeddings = LangchainEmbeddingsWrapper(embeddings)

# ==========================================
# 2. TEST DATASET (Ground Truths)
# ==========================================
# These questions are based on our actual deal documents in data/processed
eval_questions = [
    {
        "question": "What is the maturity date for the Cheeb Royalties credit agreement?",
        "ground_truth": "The maturity date for the Cheeb Royalties credit agreement is July 31, 2026."
    },
    {
        "question": "What is the maximum amount of the standby line of credit for Amerigo 2015?",
        "ground_truth": "The maximum amount of the standby line of credit is $70,000,000."
    },
    {
        "question": "Is CORRA mentioned in the Bank of Canada loan provisions?",
        "ground_truth": "Yes, the Bank of Canada loan document discusses CORRA based loan provisions."
    }
]

# ==========================================
# 3. PIPELINE INTEGRATION (Data Collection)
# ==========================================
logger.info("Initializing Agent and gathering pipeline responses...")
agent = build_agent()

questions = []
answers = []
contexts = []
ground_truths = []

for item in eval_questions:
    q = item["question"]
    gt = item["ground_truth"]
    
    logger.info(f"Evaluating Question: {q}")
    
    # Invoke our LangGraph Agent
    # retry_count and chat_history are handled by the agent state machine defaults
    result = agent.invoke({"question": q, "chat_history": []})
    
    questions.append(q)
    answers.append(result.get("answer", ""))
    ground_truths.append(gt)
    
    # RAGAS expects contexts as a List[str]. 
    # Our agent returns one big context string, so we split it (assuming chunks are separated by newline/markers)
    # If we had the raw doc list, that would be better, but we'll split the string for now.
    raw_context = result.get("context", "")
    # Splitting by a marker if exists, otherwise treat as one big chunk or split by double newline
    context_list = [c.strip() for c in raw_context.split("\n\n") if c.strip()]
    contexts.append(context_list)

# Build the dataset in the format RAGAS expects
data = {
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truth": ground_truths
}
dataset = Dataset.from_dict(data)

# ==========================================
# 4. RAGAS EVALUATION
# ==========================================
logger.info("Starting RAGAS evaluation (Faithfulness & Answer Relevancy)...")

# metrics list (Instantiated objects required in newer RAGAS versions)
# metrics = [faithfulness(), answer_relevancy()]
metrics = [faithfulness, answer_relevancy]
# Run evaluation
# We pass the critic LLM and Embeddings to keep it 100% local
results = evaluate(
    dataset=dataset,
    metrics=metrics,
    llm=critic_llm,
    embeddings=critic_embeddings
)

# ==========================================
# 5. REPORTING & CSV EXPORT
# ==========================================
logger.info("Evaluation Complete. Generating Scorecard...")

# Convert results to DataFrame for display
df_results = results.to_pandas()
print("\n" + "="*50)
print("             RAGAS EVALUATION SCORECARD")
print("="*50)

# Robust column selection (Ragas sometimes renames columns)
cols_to_show = []
for col in ["question", "user_input", "faithfulness", "answer_relevancy"]:
    if col in df_results.columns:
        cols_to_show.append(col)

if cols_to_show:
    print(df_results[cols_to_show])
else:
    print(df_results)

print("="*50)

if 'faithfulness' in df_results.columns:
    print(f"MEAN FAITHFULNESS : {df_results['faithfulness'].mean():.4f}")
if 'answer_relevancy' in df_results.columns:
    print(f"MEAN RELEVANCY    : {df_results['answer_relevancy'].mean():.4f}")
print("="*50)

# Save to CSV
output_dir = PROJECT_ROOT / "data"
output_dir.mkdir(parents=True, exist_ok=True)
csv_file = output_dir / "eval_results.csv"
df_results.to_csv(csv_file, index=False)
logger.info(f"Full evaluation results saved to: {csv_file}")
