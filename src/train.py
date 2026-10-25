import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from model import TextEncoder
from text_utils import build_vocab, encode, pad_batch


ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "training_pairs.json"
MODEL_PATH = ROOT / "models" / "text_encoder.pt"


random.seed(42)
torch.manual_seed(42)


def tensorize(texts, vocab):

    sequences = [
        encode(text, vocab)
        for text in texts
    ]

    return torch.tensor(
        pad_batch(sequences),
        dtype=torch.long,
    )


def main():

    # -----------------------------------------
    # 1. Load training data
    # -----------------------------------------

    with open(DATA_PATH) as f:
        data = json.load(f)

    documents = data["documents"]
    pairs = data["training_pairs"]


    # -----------------------------------------
    # 2. Create document lookup
    # -----------------------------------------

    document_by_id = {
        document["id"]: document["text"]
        for document in documents
    }


    # -----------------------------------------
    # 3. Build vocabulary
    # -----------------------------------------

    all_texts = (
        [doc["text"] for doc in documents]
        +
        [pair["query"] for pair in pairs]
    )

    vocab = build_vocab(all_texts)

    print("Vocabulary size:", len(vocab))

# ---- print token and IDs produced by tokenizer for model vocab
    print("\nVocabulary (token → ID):  ------- ")
    for token, token_id in vocab.items():
     print(f"{token:20s} -> {token_id}")


    # -----------------------------------------
    # 4. Create neural network
    # -----------------------------------------

    model = TextEncoder(
        vocab_size=len(vocab),
        embedding_dim=32,
        output_dim=16,
    )


    # -----------------------------------------
    # 5. Create optimizer
    # -----------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.01,
    )


    # -----------------------------------------
    # 6. Prepare queries
    # -----------------------------------------

    query_texts = [
        pair["query"]
        for pair in pairs
    ]


    # -----------------------------------------
    # 7. Get corresponding positive documents
    # -----------------------------------------

    positive_texts = [
        document_by_id[pair["document_id"]]
        for pair in pairs
    ]


    # -----------------------------------------
    # 8. Convert text → token IDs → tensors
    # -----------------------------------------

    query_ids = tensorize(
        query_texts,
        vocab
    )

    positive_ids = tensorize(
        positive_texts,
        vocab
    )


    # -----------------------------------------
    # 9. Remember document IDs
    #
    # Multiple queries may point to the same
    # relevant document.
    # -----------------------------------------

    document_ids = [
        pair["document_id"]
        for pair in pairs
    ]


    # -----------------------------------------
    # 10. Build positive mask
    #
    # True means:
    #
    # query i and document j are a valid pair.
    #
    # Example:
    #
    # query 0 -> d2
    # query 1 -> d2
    #
    # Both copies of d2 should be considered
    # positive for those queries.
    # -----------------------------------------

    positive_mask = torch.tensor(
        [
            [
                left == right
                for right in document_ids
            ]
            for left in document_ids
        ],
        dtype=torch.bool,
    )


    # -----------------------------------------
    # 11. Save INITIAL embedding weights
    #
    # We will compare these against the
    # weights after training.
    # -----------------------------------------

    initial_embeddings = (
        model.embedding.weight
        .detach()
        .clone()
    )


    # -----------------------------------------
    # 12. Training configuration
    # -----------------------------------------

    epochs = 300

    temperature = 0.1


    # =========================================
    # 13. TRAINING LOOP
    # =========================================

    for epoch in range(epochs):

        model.train()

        # Clear gradients from previous iteration
        optimizer.zero_grad()


        # -------------------------------------
        # FORWARD PASS
        # -------------------------------------

        query_vectors = model(query_ids)

        document_vectors = model(
            positive_ids
        )


        # -------------------------------------
        # Calculate cosine similarities
        #
        # query_vectors:
        #     [18, 16]
        #
        # document_vectors:
        #     [18, 16]
        #
        # similarity:
        #     [18, 18]
        # -------------------------------------

        similarity = (
            query_vectors
            @
            document_vectors.T
        ) / temperature


        # -------------------------------------
        # MULTI-POSITIVE CONTRASTIVE LOSS
        # -------------------------------------

        log_probs = F.log_softmax(
            similarity,
            dim=1
        )


        loss_per_query = -(
            log_probs
            *
            positive_mask.float()
        ).sum(dim=1) / positive_mask.sum(dim=1)


        loss = loss_per_query.mean()


        # -------------------------------------
        # BACKPROPAGATION
        # -------------------------------------

        loss.backward()


        # -------------------------------------
        # UPDATE MODEL WEIGHTS
        # -------------------------------------

        optimizer.step()


        # -------------------------------------
        # Print progress
        # -------------------------------------

        if (epoch + 1) % 50 == 0:

            print(
                f"Epoch {epoch + 1:3d}"
                f" | Loss: {loss.item():.4f}"
            )


    # =========================================
    # TRAINING FINISHED
    # =========================================


    # -----------------------------------------
    # 14. Save trained model
    # -----------------------------------------

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    torch.save(
        {
            "model_state":
                model.state_dict(),

            "vocab":
                vocab,

            "embedding_dim":
                32,

            "output_dim":
                16,

            "initial_embeddings":
                initial_embeddings,
        },
        MODEL_PATH,
    )


    print(
        "\nModel saved to:",
        MODEL_PATH
    )


    # -----------------------------------------
    # 15. Measure how much weights changed
    # -----------------------------------------

    trained_embeddings = (
        model.embedding.weight.detach()
    )


    changed = (
        trained_embeddings
        -
        initial_embeddings
    ).abs().mean()


    print(
        "Mean absolute embedding-weight change:",
        changed.item()
    )


if __name__ == "__main__":
    main()