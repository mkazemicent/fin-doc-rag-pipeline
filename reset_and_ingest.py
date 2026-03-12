import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from src.rag.chroma_deal_store import ChromaDealStore
store = ChromaDealStore()
store.reset_collection()
print("Collection and tracker reset.")
store.initialize_deal_store(access_group='general')
