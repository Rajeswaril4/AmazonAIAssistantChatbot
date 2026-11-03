
import os
import shutil
import chromadb

SOURCE_DB = "db"
TARGET_DB = "db_deploy"

# Keep the original database safe.
if os.path.exists(TARGET_DB):
    raise FileExistsError(
        f"{TARGET_DB} already exists. Rename or back it up before rerunning."
    )

source = chromadb.PersistentClient(path=SOURCE_DB)
target = chromadb.PersistentClient(path=TARGET_DB)

# Start with 1,000 products as a small deployment test.
MAX_PRODUCTS = 1000
BATCH_SIZE = 100

products_src = source.get_collection("products")
reviews_src = source.get_collection("reviews")
policy_src = source.get_collection("policy")

# Select the first 1,000 products and preserve their stored embeddings.
selected = products_src.get(
    limit=MAX_PRODUCTS,
    include=["documents", "metadatas", "embeddings"],
)

product_ids = selected["ids"]
product_asins = set(product_ids)

products_dst = target.get_or_create_collection("products")
reviews_dst = target.get_or_create_collection("reviews")
policy_dst = target.get_or_create_collection("policy")


def copy_records(src_data, dst_collection, batch_size=100):
    total = len(src_data["ids"])

    for start in range(0, total, batch_size):
        end = start + batch_size
        batch = {
            "ids": src_data["ids"][start:end],
            "metadatas": (
                src_data["metadatas"][start:end]
                if src_data["metadatas"] is not None else None
            ),
            "documents": (
                src_data["documents"][start:end]
                if src_data["documents"] is not None else None
            ),
            "embeddings": (
                src_data["embeddings"][start:end].tolist()
                if src_data["embeddings"] is not None else None
            ),
        }

        batch = {key: value for key, value in batch.items()
                 if value is not None}
        dst_collection.upsert(**batch)


copy_records(selected, products_dst, BATCH_SIZE)

# Copy only reviews whose ASIN belongs to a selected product.
# Paginate through the review collection instead of loading all reviews at once.
offset = 0
matched_reviews = 0

while True:
    batch = reviews_src.get(
        limit=BATCH_SIZE,
        offset=offset,
        include=["documents", "metadatas", "embeddings"],
    )

    if not batch["ids"]:
        break

    keep_indices = [
        i for i, metadata in enumerate(batch["metadatas"])
        if metadata and metadata.get("asin") in product_asins
    ]

    if keep_indices:
        filtered = {
            "ids": [batch["ids"][i] for i in keep_indices],
            "metadatas": [batch["metadatas"][i] for i in keep_indices],
        }

        if batch["documents"] is not None:
            filtered["documents"] = [
                batch["documents"][i] for i in keep_indices
            ]

        if batch["embeddings"] is not None:
            filtered["embeddings"] = [
                batch["embeddings"][i].tolist() for i in keep_indices
            ]

        reviews_dst.upsert(**filtered)
        matched_reviews += len(keep_indices)

    offset += len(batch["ids"])

    if offset % 50000 == 0:
        print(f"Scanned {offset:,} reviews; matched {matched_reviews:,}")

# Copy every policy record, including existing embeddings.
policy_data = policy_src.get(
    include=["documents", "metadatas", "embeddings"]
)
copy_records(policy_data, policy_dst, BATCH_SIZE)

print("\nDeployment database created.")
print("Products:", products_dst.count())
print("Matching reviews:", reviews_dst.count())
print("Policy records:", policy_dst.count())
print("Database path:", os.path.abspath(TARGET_DB))

