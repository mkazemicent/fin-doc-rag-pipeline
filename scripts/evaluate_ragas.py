import sys
import json
import logging
from pathlib import Path
from datasets import Dataset

# --- PATH PRE-REQUISITE ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Local Imports
try:
    from src.config import get_settings
    from src.rag.deal_analyzer import build_deal_analyzer
    from langchain_ollama import ChatOllama, OllamaEmbeddings
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
except ImportError as e:
    print("Error: Missing dependencies. Please run: pip install ragas datasets")
    print(f"Details: {e}")
    sys.exit(1)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    settings = get_settings()

    # ==========================================
    # 1. LOCAL MODEL WRAPPERS
    # ==========================================
    logger.info("Configuring local Ollama models for RAGAS evaluation...")

    # Critic/Judge LLM
    llm = ChatOllama(
        model=settings.llm_model,
        base_url=settings.ollama_base_url,
        temperature=0
    )
    critic_llm = LangchainLLMWrapper(llm)

    # Evaluator Embeddings
    embeddings = OllamaEmbeddings(
        model=settings.embedding_model,
        base_url=settings.ollama_base_url
    )
    critic_embeddings = LangchainEmbeddingsWrapper(embeddings)

    # ==========================================
    # 2. TEST DATASET (Ground Truths)
    # ==========================================
    # Load evaluation questions from external JSON file
    eval_dataset_path = Path(__file__).parent / "eval_dataset.json"
    with open(eval_dataset_path) as f:
        eval_questions = json.load(f)

    # ==========================================
    # 3. PIPELINE INTEGRATION (Data Collection)
    # ==========================================
    logger.info("Initializing Agent and gathering pipeline responses...")
    agent = build_deal_analyzer()

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
        # Use the raw Document list from AgentState for clean context without header noise.
        retrieved_docs = result.get("retrieved_docs", [])
        context_list = [doc.page_content for doc in retrieved_docs]
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
    output_dir = settings.data_root
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_file = output_dir / "eval_results.csv"
    df_results.to_csv(csv_file, index=False)
    logger.info(f"Full evaluation results saved to: {csv_file}")
