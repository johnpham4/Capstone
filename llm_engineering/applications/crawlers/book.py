import fitz
import os
from typing import List
from llm_engineering.applications.crawlers.base import BaseCrawler
from llm_engineering.applications.networks.yolo_layout import YOLOLayoutDetector
from llm_engineering.applications.preprocessing.pdf_page_extractor import PDFPageExtractor
from llm_engineering.applications.preprocessing.ocr_extractor import DocTRExtractor
from llm_engineering.domains.documents import TextbookPageDocument
from llm_engineering.infrastructures.db.mongo import MongoConnection


class BookCrawler(BaseCrawler):
    """
    Crawl PDF textbook page by page
    1 page → 1 MongoDB document (text + images)
    """

    def extract(self, pdf_path: str, **kwargs) -> List[TextbookPageDocument]:
        """
        Extract all pages from PDF

        Args:
            pdf_path: Path to PDF file
            **kwargs:
                - output_dir: Temp directory (default: ./tmp)
                - save_to_db: Save to MongoDB (default: True)

        Returns:
            List of TextbookPageDocument
        """
        output_dir = kwargs.get("output_dir", "./tmp")
        save_to_db = kwargs.get("save_to_db", True)

        os.makedirs(output_dir, exist_ok=True)

        # Get total pages
        doc = fitz.open(pdf_path)
        total_pages = doc.page_count
        doc.close()

        print(f"📄 PDF: {pdf_path}")
        print(f"📊 Total pages: {total_pages}")
        print("-" * 60)

        # Initialize models (Singleton)
        yolo = YOLOLayoutDetector()
        ocr = DocTRExtractor()

        all_documents = []

        for page_num in range(total_pages):
            print(f"[{page_num + 1}/{total_pages}] Processing page {page_num}...")

            try:
                doc = self._extract_page(
                    pdf_path=pdf_path,
                    page_num=page_num,
                    yolo_model=yolo,
                    ocr_model=ocr,
                    output_dir=output_dir
                )

                all_documents.append(doc)

                if save_to_db:
                    self._save_to_mongo(doc)

                print(f"  ✓ Text: {len(doc.full_text)} chars, Images: {len(doc.images)}")

            except Exception as e:
                print(f"  ✗ Error: {e}")
                continue

        print("\n" + "=" * 60)
        print(f"✅ Extracted {len(all_documents)}/{total_pages} pages")

        # Cleanup
        self._cleanup(output_dir)

        return all_documents

    def _extract_page(self, pdf_path: str, page_num: int, yolo_model, ocr_model, output_dir: str) -> TextbookPageDocument:
        """Extract single page"""
        extractor = PDFPageExtractor(pdf_path=pdf_path, page_num=page_num)

        # Step 1: Render page to image
        page_image_path = extractor.render_page_to_image(output_dir)

        # Step 2: YOLO detect layout blocks
        blocks = extractor.detect_blocks_with_yolo(yolo_model, page_image_path)

        # Step 3: Crop and extract blocks
        page_blocks = extractor.crop_and_extract_blocks(blocks, output_dir)

        # Step 4: Separate text and images
        full_text = ""
        images = []

        for block in page_blocks:
            if block.block_type == "picture":
                # Save image as base64
                images.append(block.content)
            else:
                # OCR text blocks
                if os.path.exists(block.content):
                    text = ocr_model.extract_text(block.content)
                    full_text += text + "\n"

        # Step 5: Create document
        doc = TextbookPageDocument(
            page_number=page_num,
            source_pdf=os.path.basename(pdf_path),
            full_text=full_text.strip(),
            images=images
        )

        return doc

    def _save_to_mongo(self, doc: TextbookPageDocument):
        """Save document to MongoDB"""
        mongo = MongoConnection()
        db = mongo[self._get_database_name()]
        collection = db[TextbookPageDocument.Settings.name]
        collection.insert_one(doc.dict())

    def _get_database_name(self) -> str:
        """Get database name from settings"""
        from llm_engineering.settings import settings
        return settings.MONGO_DATABASE

    def _cleanup(self, output_dir: str):
        """Clean up temporary files"""
        import shutil
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
            print(f"🗑️  Cleaned up: {output_dir}")