# services/objects.py
import random

def _metadata_to_object(metadata):
    """Convert a Chroma metadata dict into our object shape."""
    return {
        "museum_number": metadata.get("museum_number"),
        "object_name": metadata.get("object_name", "Unknown object"),
        "object_type": metadata.get("object_type", ""),
        "culture": metadata.get("culture", ""),
        "production_date": metadata.get("production_date", ""),
        "materials": metadata.get("materials", ""),
        "location": metadata.get("location", ""),
        "image": metadata.get("image", ""),
    }


class ObjectsService:
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore

    def get_random(self):
        all_items = self.vectorstore._collection.get(include=["metadatas"])
        metadatas = all_items["metadatas"]

        unique_objects = {}
        for metadata in metadatas:
            museum_number = metadata.get("museum_number")
            if museum_number and museum_number not in unique_objects:
                unique_objects[museum_number] = _metadata_to_object(metadata)

        return random.choice(list(unique_objects.values()))

    def get_by_museum_number(self, museum_number):
        results = self.vectorstore._collection.get(
            where={"museum_number": museum_number},
            include=["metadatas"],
            limit=1,
        )
        if not results["metadatas"]:
            return None
        return _metadata_to_object(results["metadatas"][0])