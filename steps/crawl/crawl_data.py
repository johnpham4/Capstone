# from typing import Annotated, List, Dict, Any
# from zenml import step
# from tqdm.auto import tqdm
# from llm_engineering.applications.crawlers.dispatcher import CrawlerDispatcher
# from loguru import logger


# @step
# def crawl_links(
#     pdf_sources: List[Dict[str, Any]] = []
# ) -> Annotated[int, "num_crawled"]:

#     if not pdf_sources:
#         logger.warning("No PDF sources provided")
#         return 0

#     dispatcher = CrawlerDispatcher.build().register_pdf()

#     success_count = 0
#     for source in tqdm(pdf_sources, desc="Extracting figures from PDFs"):
#         try:
#             url = source.get("url")
#             name = source.get("name", "unknown")
#             output_dir = f"dataset/geometry_figures/{name}"
#             doc_type = source.get("type", "pdf")  # Default to pdf
#             has_dot_str = source.get("has_dot", "True")  # Get as string
#             has_dot = has_dot_str == "True"  # Convert to boolean

#             logger.info(f"Processing [{name}] ({doc_type}, has_dot={has_dot}): {url}")

#             # Get crawler by type instead of URL extension
#             if doc_type == "pdf":
#                 crawler = dispatcher._crawlers.get('pdf')
#             elif doc_type == "docx":
#                 crawler = dispatcher._crawlers.get('docx')
#             else:
#                 logger.error(f"Unknown document type: {doc_type}")
#                 continue

#             crawler.extract(link=url, output_dir=output_dir, name=name, has_dot=has_dot)
#             success_count += 1

#         except Exception as e:
#             logger.error(f"Failed to extract figures from {source.get('name', 'unknown')}: {e}")

#     logger.info(f"Successfully processed {success_count}/{len(pdf_sources)} PDFs")
#     return success_count
