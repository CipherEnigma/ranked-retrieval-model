class Document:
    def __init__(self, docid, category, title, text):
        self.docid = docid
        self.category = category
        self.title = title
        self.text = text
        self.tokens = []

    def __repr__(self):
        return (
            f"Document("
            f"docid='{self.docid}', "
            f"category='{self.category}', "
            f"title='{self.title}'"
            f")"
        )