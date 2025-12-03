import faiss
import numpy as np
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
import time

# --- 설정 ---
MONGO_URI = "mongodb://localhost:27017/"
INDEX_FILE = "large_index.faiss"

# 1. MongoDB 연결
client = MongoClient(MONGO_URI)
collection = client["bigdata_db"]["news"]

# 2. 모델 로드
model = SentenceTransformer('all-MiniLM-L6-v2')

# 3. FAISS 인덱스 로드 (Memory Mapping 핵심!)
print(">>> 인덱스 로딩 중 (MMAP 모드)...")
# IO_FLAG_MMAP: 파일을 RAM에 다 올리지 않고, 필요할 때 디스크에서 읽음
index = faiss.read_index(INDEX_FILE, faiss.IO_FLAG_MMAP)
print(f">>> 로드 완료. 총 데이터 수: {index.ntotal}")

# nprobe 설정: 몇 개의 클러스터(방)를 뒤질 것인가?
# 값이 높으면 정확도 상승, 속도 저하. (보통 nlist의 5~10% 설정)
index.nprobe = 10 

def search(query, k=3):
    start_time = time.time()
    
    # (1) 쿼리 벡터 변환
    q_vec = model.encode([query]).astype('float32')
    
    # (2) FAISS 검색 (디스크 I/O 발생)
    # D: 거리(Distance), I: 인덱스(ID)
    D, I = index.search(q_vec, k)
    
    search_time = time.time() - start_time
    
    # (3) 결과 매핑 (FAISS ID -> MongoDB 조회)
    found_ids = I[0].tolist() # 예: [105, 5002, 12]
    distances = D[0].tolist()
    
    print(f"🔎 검색어: '{query}' (소요시간: {search_time:.4f}초)")
    print("-" * 50)
    
    if found_ids[0] == -1:
        print("결과 없음.")
        return

    # MongoDB에서 uid 리스트로 한 번에 조회 ($in 연산자 사용)
    # 인덱스("uid")가 걸려있어 매우 빠름
    cursor = collection.find({"uid": {"$in": found_ids}})
    
    # 결과를 딕셔너리로 변환하여 순서 맞추기
    mongo_docs = {doc["uid"]: doc for doc in cursor}
    
    for i, uid in enumerate(found_ids):
        if uid in mongo_docs:
            doc = mongo_docs[uid]
            print(f"[{i+1}위] UID: {doc['uid']} | 유사도 거리: {distances[i]:.4f}")
            print(f"제목: {doc['title']}")
            print(f"내용: {doc['content'][:50]}...") # 내용 미리보기
            print("")
        else:
            print(f"[{i+1}위] MongoDB에서 문서(UID:{uid})를 찾을 수 없음.")

# --- 실행 ---
while True:
    q = input("\n검색어를 입력하세요 (종료: q): ")
    if q == 'q': break
    search(q)
