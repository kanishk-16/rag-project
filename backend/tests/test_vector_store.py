import numpy as np
from app.retrieval.vector_store import _min_max_normalize, store_chunks, retrieve_top_k

def test_min_max_normalize():
    # Regular values
    scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    norm = _min_max_normalize(scores)
    assert np.isclose(norm[0], 0.0)
    assert np.isclose(norm[-1], 1.0)

    # Identical values should return zeros (no division by zero)
    same_scores = np.array([5.0, 5.0, 5.0], dtype=np.float32)
    norm_same = _min_max_normalize(same_scores)
    assert np.all(norm_same == 0.0)

    # Empty array
    assert len(_min_max_normalize(np.empty(0))) == 0


def test_user_isolation_retrieval():
    # Store chunks for user_alice
    doc_alice = "doc_alice_1"
    chunks_alice = [{"text": "Alice secret password is blue-rabbit", "page_num": 1}]
    # 384-dim dummy embeddings
    emb_alice = [np.ones(384, dtype=np.float32).tolist()]
    store_chunks(chunks_alice, emb_alice, doc_id=doc_alice, user_id="user_alice")

    # Store chunks for user_bob
    doc_bob = "doc_bob_1"
    chunks_bob = [{"text": "Bob favorite fruit is mango", "page_num": 1}]
    emb_bob = [np.zeros(384, dtype=np.float32).tolist()]
    store_chunks(chunks_bob, emb_bob, doc_id=doc_bob, user_id="user_bob")

    dummy_query_emb = np.ones(384, dtype=np.float32).tolist()

    # Query as Bob for Alice's secret
    bob_results = retrieve_top_k(
        dummy_query_emb,
        top_k=5,
        query_text="blue-rabbit",
        user_id="user_bob"
    )
    # Bob should NOT retrieve Alice's chunk
    assert all(r["metadata"]["user_id"] == "user_bob" for r in bob_results)
    assert not any("blue-rabbit" in r["text"] for r in bob_results)

    # Query as Alice for Alice's secret
    alice_results = retrieve_top_k(
        dummy_query_emb,
        top_k=5,
        query_text="blue-rabbit",
        user_id="user_alice"
    )
    assert any("blue-rabbit" in r["text"] for r in alice_results)
    assert all(r["metadata"]["user_id"] == "user_alice" for r in alice_results)
