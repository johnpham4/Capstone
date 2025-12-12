from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from llm_engineering.applications.networks.base import SingletonMeta


class DocTRExtractor(metaclass=SingletonMeta):
    """DocTR OCR model - Singleton"""

    def __init__(self):
        self.model = ocr_predictor(pretrained=True)

    def extract_text(self, image_path: str) -> str:
        """Extract text from image using OCR"""
        doc = DocumentFile.from_images(image_path)
        result = self.model(doc)

        text_content = ""
        for page in result.pages:
            for block in page.blocks:
                for line in block.lines:
                    for word in line.words:
                        text_content += word.value + " "
                    text_content += "\n"

        return text_content.strip()
