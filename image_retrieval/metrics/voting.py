import os

def compute_voting_precision(ranks, scores, q_paths, imlist, image_to_place, top_k=5):
    """
    Calcula precisión basada en votación ponderada por score sobre el top-k resultados, con prints para depuración.
    """
    nq = ranks.shape[1]
    aciertos = 0

    for i in range(nq):
        query_name = os.path.basename(q_paths[i])
        query_place = image_to_place.get(query_name, None)

        if query_place is None:
            continue

        place_scores = {}

        for j in range(top_k):
            db_idx = ranks[j, i]
            db_name = imlist[db_idx] + '.jpg'
            pred_place = image_to_place.get(db_name, None)
            score = scores[j, i]

            if pred_place is not None:
                place_scores[pred_place] = place_scores.get(pred_place, 0) + score

        if not place_scores:
            continue

        predicted_place = max(place_scores.items(), key=lambda x: x[1])[0]

        if predicted_place == query_place:
            aciertos += 1

    precision = aciertos / nq
    return precision