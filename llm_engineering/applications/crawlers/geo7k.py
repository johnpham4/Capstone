import subprocess
import os
import tempfile
import shutil
from loguru import logger
import base64
import json

from llm_engineering.applications.crawlers.base import BaseCrawler
from llm_engineering.domains.documents import Geo7kDocument


class Geo7kCrawler(BaseCrawler):
    model = Geo7kDocument
    link: str = "https://github.com/BitSecret/formalgeo7k.git"

    def extract(self) -> None:
        if not self.link:
            raise ValueError("link must not be None")

        logger.info(f"Starting scraping GitHub repository: {self.link}")

        repo_name = self.link.rstrip("/").split("/")[-1]
        local_temp = tempfile.mkdtemp(prefix="geo7k_")

        try:
            subprocess.run(
                ["git", "clone", self.link, local_temp],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            logger.info("Repository cloned successfully")

            problem_dir = os.path.join(local_temp, "formalgeo7k_v1", "problems")
            diagram_dir = os.path.join(local_temp, "formalgeo7k_v1", "diagrams")

            if not os.path.exists(problem_dir):
                raise RuntimeError("Dataset structure mismatch: problems folder not found")

            json_files = [f for f in os.listdir(problem_dir) if f.endswith(".json")]

            for jf in json_files:
                jf_path = os.path.join(problem_dir, jf)
                with open(jf_path, "r", encoding="utf-8") as f:
                    content = json.load(f.read())

                base = jf.split(".")[0]
                img_path = os.path.join(diagram_dir, base + ".png")
                
                if not img_path:
                    raise RuntimeError(f"No available image for problems {base}")
                    
                with open(img_path, "rb") as f:
                    encoded = base64.b64encode(f.read()).decode("utf-8")
                    
                problem = content[""]

                instance = self.model(
                    problem=problem,
                    image=encoded,
                    lan="vi",
                    text_cdl=""
                )
                instance.save()

                logger.debug(f"Saved JSON: {jf_path}")

        except Exception as e:
            logger.error(f"Error while processing GitHub clone: {e}")
            raise

        finally:
            shutil.rmtree(local_temp)
            logger.info("Temporary directory cleaned up")

        logger.info(f"Finished scraping GitHub repository: {self.link}")
