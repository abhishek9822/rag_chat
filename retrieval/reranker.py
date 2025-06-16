def reciprocal_rag_fusion(results:list[list],k= 60):
    fused_scores = {}
    documents = {}
    for docs in results:
        for rank,doc in enumerate(docs):
            doc_str = doc.page_content
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
                documents[doc_str] = doc

            fused_scores[doc_str] += 1 / (rank + k)

    reranked_doc_strs = sorted(
        fused_scores,key=lambda d: fused_scores[d],reverse=True)
    return [documents[doc_str] for doc_str in reranked_doc_strs]