import os
import chromadb

SOURCE_DB = "db"
TARGET_DB = "db_deploy"

# Keep the original database safe. Back up or rename db_deploy before rerunning.
if os.path.exists(TARGET_DB):
    raise FileExistsError(
        f"{TARGET_DB} already exists. Rename it to db_deploy_backup first."
    )

MAX_PRODUCTS = 1000
BATCH_SIZE = 100

source = chromadb.PersistentClient(path=SOURCE_DB)
target = chromadb.PersistentClient(path=TARGET_DB)

products_src = source.get_collection("products")
reviews_src = source.get_collection("reviews")
policy_src = source.get_collection("policy")

# Discover the categories present in the original product collection.
all_metadata = products_src.get(include=["metadatas"])["metadatas"]
categories = sorted({
    metadata.get("category")
    for metadata in all_metadata
    if metadata and metadata.get("category")
})

if not categories:
    raise RuntimeError("No product categories were found in the source database.")

print("Source categories:", categories)

# Select products across categories instead of taking the first 1,000 records,
# which previously selected only Handmade_Products.
per_category = max(1, MAX_PRODUCTS // len(categories))
selected_ids = []
selected_metadatas = []
selected_documents = []
selected_embeddings = []

for category in categories:
    result = products_src.get(
        where={"category": category},
        limit=per_category,
        include=["documents", "metadatas", "embeddings"],
    )
    selected_ids.extend(result["ids"])
    selected_metadatas.extend(result["metadatas"] or [])
    if result["documents"] is not None:
        selected_documents.extend(result["documents"])
    if result["embeddings"] is not None:
        selected_embeddings.extend(result["embeddings"])

# If the total is below the requested limit because of uneven category sizes,
# fill from any products not already selected.
if len(selected_ids) < MAX_PRODUCTS:
    selected_set = set(selected_ids)
    remaining = products_src.get(
        limit=products_src.count(),
        include=["documents", "metadatas", "embeddings"],
    )
    for i, product_id in enumerate(remaining["ids"]):
        if product_id in selected_set:
            continue
        selected_ids.append(product_id)
        selected_metadatas.append(remaining["metadatas"][i])
        if remaining["documents"] is not None:
            selected_documents.append(remaining["documents"][i])
        if remaining["embeddings"] is not None:
            selected_embeddings.append(remaining["embeddings"][i])
        if len(selected_ids) >= MAX_PRODUCTS:
            break

# Keep all arrays aligned if fewer than MAX_PRODUCTS were available.
selected = {
    "ids": selected_ids[:MAX_PRODUCTS],
    "metadatas": selected_metadatas[:MAX_PRODUCTS],
}
if selected_documents:
    selected["documents"] = selected_documents[:MAX_PRODUCTS]
if selected_embeddings:
    selected["embeddings"] = [
    e.tolist() if hasattr(e, "tolist") else e
    for e in selected_embeddings[:MAX_PRODUCTS]
]
product_asins = set(selected["ids"])

products_dst = target.get_or_create_collection("products")
reviews_dst = target.get_or_create_collection("reviews")
policy_dst = target.get_or_create_collection("policy")


def copy_records(src_data, dst_collection, batch_size=100):
    total = len(src_data["ids"])
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = {"ids": src_data["ids"][start:end]}
        for key in ("metadatas", "documents", "embeddings"):
            if key in src_data and src_data[key] is not None:
                values = src_data[key][start:end]
                if key == "embeddings":
                    values = values.tolist() if hasattr(values, "tolist") else [
                        v.tolist() if hasattr(v, "tolist") else v for v in values
                    ]
                batch[key] = values
        dst_collection.upsert(**batch)


copy_records(selected, products_dst, BATCH_SIZE)

# Copy only reviews associated with selected products.
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
        i for i, metadata in enumerate(batch["metadatas"] or [])
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
            embeddings = [batch["embeddings"][i] for i in keep_indices]
            filtered["embeddings"] = [
                e.tolist() if hasattr(e, "tolist") else e for e in embeddings
            ]
        reviews_dst.upsert(**filtered)
        matched_reviews += len(keep_indices)

    offset += len(batch["ids"])
    if offset % 50000 == 0:
        print(f"Scanned {offset:,} reviews; matched {matched_reviews:,}")

policy_data = policy_src.get(
    include=["documents", "metadatas", "embeddings"]
)
copy_records(policy_data, policy_dst, BATCH_SIZE)

from collections import Counter
final_metadata = products_dst.get(include=["metadatas"])["metadatas"]
print("\nDeployment database created.")
print("Products:", products_dst.count())
print("Product categories:", Counter(
    str(m.get("category", "Unknown")) for m in final_metadata
))
print("Matching reviews:", reviews_dst.count())
print("Policy records:", policy_dst.count())
print("Database path:", os.path.abspath(TARGET_DB))
