"""
Generates the sample resume corpus in data/resumes/.

Run once: `python scripts/make_sample_resumes.py`. The generated files are
committed to the repo, so reviewers never need to run this. It exists so the
corpus is reproducible and so the format mix (TXT / PDF / DOCX) is explicit.

The twelve candidates are written to span the decision space: clear hires,
clear rejects, and -- most usefully -- several genuinely ambiguous profiles
(adjacent-domain experts, an over-qualified senior, a strong fresher) where a
naive keyword matcher and a good screener disagree.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OUT = Path(__file__).resolve().parent.parent / "data" / "resumes"

RESUMES: list[tuple[str, str, str]] = []


def add(filename: str, fmt: str, body: str) -> None:
    RESUMES.append((filename, fmt, body.strip()))


add("priya_raghavan.pdf", "pdf", """
Priya Raghavan
Bengaluru, India | priya.raghavan@email.com | +91 98450 11234
github.com/praghavan | linkedin.com/in/priyaraghavan

SUMMARY
Machine Learning Engineer with 5 years building and shipping production NLP
systems. Owns models end to end from data curation to deployment and monitoring.

EXPERIENCE

Senior ML Engineer, Quantiva Systems, Bengaluru
2023 - Present
- Built a document classification service (BERT fine-tuned in PyTorch) serving
  4M documents/month; raised macro-F1 from 0.71 to 0.89.
- Designed a retrieval-augmented generation pipeline over 12M customer
  documents using FAISS and a HuggingFace bi-encoder; cut support handling
  time 34%.
- Containerised all models with Docker and deployed behind FastAPI endpoints
  on AWS ECS; added drift monitoring with MLflow and Prometheus.
- Mentored two junior engineers and ran the team's code review rota.

ML Engineer, Trailhead Analytics, Pune
2021 - 2023
- Trained NER models for invoice field extraction using spaCy and PyTorch.
- Built ETL pipelines in Python and SQL over Postgres; automated with Airflow.
- Wrote pytest suites covering data validation and model regression checks.

EDUCATION
M.Tech in Computer Science, IIT Madras, 2021
B.E. in Information Science, RV College of Engineering, 2019

SKILLS
Python, PyTorch, HuggingFace Transformers, spaCy, scikit-learn, FAISS, SQL,
Postgres, Docker, FastAPI, AWS (ECS, S3, Lambda), MLflow, Airflow, Git,
GitHub Actions, pytest, Linux
""")

add("arjun_menon.docx", "docx", """
Arjun Menon
Kochi, India | arjun.menon@email.com | +91 99470 55321

PROFILE
NLP engineer with 4 years of experience, focused on retrieval systems and
large language model applications in production.

WORK EXPERIENCE

NLP Engineer, Fluxwave AI
2022 - Present
- Owns the company's RAG stack: chunking, embedding, Pinecone vector database,
  reranking and answer synthesis. Serves 800k queries per month.
- Fine-tuned open-weight LLMs with LoRA in PyTorch for domain question
  answering; reduced hallucination rate on the internal eval set by 41%.
- Shipped models as Dockerised REST services; CI/CD via GitHub Actions.
- Built evaluation harnesses so every model change is measured before release.

Junior Data Scientist, Meridian Retail
2021 - 2022
- Text classification for customer support ticket routing (scikit-learn, then
  DistilBERT). Wrote SQL to build the training set from the warehouse.

EDUCATION
B.Tech in Computer Science and Engineering, NIT Calicut, 2021

TECHNICAL SKILLS
Python | PyTorch | HuggingFace Transformers | LangChain | Pinecone | FAISS |
RAG | SQL | Docker | FastAPI | Git | AWS S3 and Lambda | Weights & Biases
""")

add("elena_petrova.pdf", "pdf", """
Elena Petrova
Berlin, Germany | elena.petrova@email.com | +49 151 2233 4455

SUMMARY
Principal Machine Learning Engineer, 9 years of experience across NLP,
recommendation systems and large-scale ML platforms. Has led teams of six.

EXPERIENCE

Principal ML Engineer, Hanseatic Data GmbH
2020 - Present
- Led the NLP platform team; owned multilingual text classification and
  entity linking serving 40M requests/day.
- Architected the company's feature store and model serving layer on
  Kubernetes; standardised deployment for 30+ models.
- Built distributed training on Spark and PyTorch DDP across 64 GPUs.
- Set the org's model governance and monitoring standards.

Senior ML Engineer, Nordlicht Commerce
2017 - 2020
- Recommendation systems using collaborative filtering and neural rankers.
- Owned the A/B testing framework and statistical analysis of experiments.

EDUCATION
M.Sc. in Computer Science, TU Munich, 2017
B.Sc. in Mathematics, Humboldt University, 2015

SKILLS
Python, PyTorch, TensorFlow, Transformers, NLP, Spark, Kubernetes, Docker,
AWS, GCP, SQL, Airflow, MLflow, Kubeflow, Git, Scala, Linux, A/B testing
""")

add("meera_nair.txt", "txt", """
Meera Nair
Chennai, India | meera.nair@email.com | +91 90030 77812

OBJECTIVE
NLP engineer with 3 years of hands-on experience looking to work on applied
language systems at scale.

EXPERIENCE

Machine Learning Engineer,Avartan Labs
2023 - Present
- Fine-tuned transformer models (BERT, RoBERTa) in PyTorch for legal document
  classification across seven document types.
- Built a named entity recognition pipeline for contract clause extraction.
- Deployed models with Docker and Flask; wrote the REST API layer.
- Queried and shaped training data with SQL on a Postgres warehouse.

Data Science Intern, Avartan Labs
2022 - 2023
- Exploratory analysis and baseline models with scikit-learn and pandas.

PROJECTS
- Open-source Tamil text normalisation library, 300+ GitHub stars.
- Kaggle: top 8% in a toxic comment classification competition.

EDUCATION
B.Tech in Artificial Intelligence and Data Science, Anna University, 2022

SKILLS
Python, PyTorch, Transformers, HuggingFace, NLP, NER, scikit-learn, pandas,
NumPy, SQL, Postgres, Docker, Flask, Git, Linux, matplotlib
""")

add("daniel_okafor.docx", "docx", """
Daniel Okafor
Lagos, Nigeria | daniel.okafor@email.com | +234 802 555 1199

SUMMARY
MLOps engineer with 5 years of experience productionising machine learning.
Strongest on infrastructure, deployment, reliability and observability.

EXPERIENCE

MLOps Engineer, Kano Cloud Services
2021 - Present
- Built the model deployment platform: Docker images, Kubernetes operators,
  canary rollouts and automatic rollback for 40+ models.
- Ran MLflow and Weights & Biases for experiment tracking across four teams.
- Implemented drift and data-quality monitoring; cut silent model failures
  to near zero.
- Managed AWS infrastructure with Terraform; SageMaker for training jobs.
- Built Spark pipelines for feature generation over 3TB of daily events.

DevOps Engineer, Sahara Payments
2019 - 2021
- CI/CD with Jenkins and GitHub Actions; Linux administration; Postgres.

EDUCATION
B.Sc. in Computer Science, University of Lagos, 2019

SKILLS
Python, Docker, Kubernetes, AWS, SageMaker, Terraform, MLflow, Kubeflow,
Weights & Biases, Spark, Airflow, SQL, Postgres, Jenkins, GitHub Actions,
Git, Linux, Bash, FastAPI, some PyTorch
""")

add("wei_zhang.pdf", "pdf", """
Wei Zhang
Singapore | wei.zhang@email.com | +65 8123 4567

RESEARCH SUMMARY
PhD in computational linguistics with 3 years of applied research experience.
Nine peer-reviewed publications in ACL, EMNLP and NAACL.

EXPERIENCE

Research Scientist, Lion City AI Lab
2023 - Present
- Research on parameter-efficient fine-tuning of large language models.
- Published three first-author papers on low-resource machine translation.
- Prototype implementations in PyTorch and HuggingFace Transformers.
- Collaborated with engineering to hand off one model to production.

Research Assistant, National University of Singapore
2019 - 2023
- Doctoral research on syntactic parsing for low-resource languages.
- Built and released two annotated corpora used by the wider community.

EDUCATION
Ph.D. in Computational Linguistics, National University of Singapore, 2023
B.A. in Linguistics, Peking University, 2018

SKILLS
Python, PyTorch, Transformers, HuggingFace, NLP, deep learning, machine
learning, LaTeX, Git, Linux, statistics, some SQL
""")

add("fatima_sheikh.txt", "txt", """
Fatima Sheikh
Karachi, Pakistan | fatima.sheikh@email.com | +92 300 1234567

SUMMARY
Computer vision engineer, 5 years, shipping deep learning models for
industrial inspection and medical imaging.

EXPERIENCE

Senior Computer Vision Engineer, Indus Vision Technologies
2022 - Present
- Trained and deployed object detection and segmentation models in PyTorch
  (YOLOv8, U-Net) for defect detection on production lines.
- Optimised inference with TensorRT and ONNX; 6x latency improvement.
- Containerised the full inference stack with Docker; deployed on edge devices
  and on AWS EC2 GPU instances behind a REST API.
- Built annotation tooling and data pipelines; SQL over Postgres.

Machine Learning Engineer, Meraj Diagnostics
2021 - 2022
- Image classification for radiology triage; scikit-learn and PyTorch.

EDUCATION
M.S. in Electrical Engineering, NUST Islamabad, 2021
B.E. in Electronics, NED University, 2019

SKILLS
Python, PyTorch, OpenCV, computer vision, deep learning, TensorFlow, ONNX,
Docker, AWS EC2, REST API, FastAPI, SQL, Git, Linux, NumPy, pandas
""")

add("sneha_kulkarni.docx", "docx", """
Sneha Kulkarni
Mumbai, India | sneha.kulkarni@email.com | +91 98200 44556

SUMMARY
Data scientist with 6 years of experience in analytics, experimentation and
business intelligence for consumer fintech.

EXPERIENCE

Lead Data Scientist, Paylane Financial
2022 - Present
- Built credit risk scoring models using scikit-learn (gradient boosting);
  improved approval rates 12% at constant default rate.
- Designed and analysed A/B tests; owns the experimentation statistics.
- Built executive dashboards in Tableau and Power BI.
- Heavy SQL over BigQuery; Python for analysis with pandas and NumPy.

Data Analyst, Suvidha Bank
2020 - 2022
- Cohort and retention analysis; time series forecasting of transaction volume.
- Automated recurring reporting with Python and Excel.

EDUCATION
M.Sc. in Statistics, University of Mumbai, 2020
B.Sc. in Mathematics, Fergusson College, 2018

SKILLS
Python, pandas, NumPy, scikit-learn, statistics, A/B testing, SQL, BigQuery,
Tableau, Power BI, Excel, time series forecasting, machine learning, Git
""")

add("karthik_reddy.txt", "txt", """
Karthik Reddy
Hyderabad, India | karthik.reddy@email.com | +91 94900 33221

SUMMARY
Data engineer with 6 years of experience building batch and streaming data
platforms.

EXPERIENCE

Senior Data Engineer, Deccan Data Works
2021 - Present
- Built and operated Spark pipelines processing 5TB/day on Databricks.
- Orchestrated 200+ Airflow DAGs; owns the data quality framework.
- Modelled the warehouse in Snowflake; extensive SQL and dbt.
- Kafka streaming ingestion; Docker and Kubernetes for pipeline services.

Data Engineer, Vindhya Systems
2020 - 2021
- ETL from Postgres and MongoDB into the analytics warehouse.

EDUCATION
B.Tech in Computer Science, JNTU Hyderabad, 2020

SKILLS
Python, Scala, Spark, PySpark, Databricks, Airflow, Kafka, SQL, Snowflake,
Postgres, MongoDB, dbt, Docker, Kubernetes, AWS S3 and EMR, Git, Linux
""")

add("ananya_iyer.pdf", "pdf", """
Ananya Iyer
Coimbatore, India | ananya.iyer@email.com | +91 98940 22110

OBJECTIVE
Recent graduate seeking a machine learning engineering role. Strong project
portfolio in NLP.

EXPERIENCE

Machine Learning Intern, Zentari Software
2025 - 2026
- Built a sentiment classification model with HuggingFace Transformers and
  PyTorch; deployed a demo with FastAPI and Docker.
- Wrote SQL queries to assemble the training dataset.

PROJECTS
- Resume-to-job matcher: TF-IDF and sentence embeddings, Streamlit front end.
- Abstractive news summariser fine-tuned from T5-small on PyTorch.
- Multilingual toxicity classifier; wrote the data collection scraper.

EDUCATION
B.E. in Computer Science, PSG College of Technology, 2026
CGPA 9.1/10. Coursework: machine learning, deep learning, NLP, databases.

SKILLS
Python, PyTorch, HuggingFace Transformers, NLP, scikit-learn, pandas, NumPy,
SQL, Docker, FastAPI, Git, Linux, matplotlib, Streamlit
""")

add("vikram_desai.docx", "docx", """
Vikram Desai
Ahmedabad, India | vikram.desai@email.com | +91 99250 66778

SUMMARY
Full stack engineer with 4 years of experience building web applications.
Interested in moving toward machine learning work.

EXPERIENCE

Full Stack Engineer, Sabarmati Softworks
2022 - Present
- Built React and TypeScript front ends against Node.js and Django REST APIs.
- Postgres schema design and query optimisation.
- Dockerised services deployed to AWS; CI with GitHub Actions.
- Integrated a third-party LLM API to add a chat assistant feature to the
  product; wrote the prompt templates and the caching layer.

Software Engineer, Riverfront Tech
2021 - 2022
- Java and Spring Boot microservices; unit testing with JUnit.

EDUCATION
B.E. in Computer Engineering, Gujarat Technological University, 2021

SKILLS
JavaScript, TypeScript, React, Node.js, Python, Django REST, Java, Spring Boot,
Postgres, SQL, Docker, AWS, Git, GitHub Actions, REST API, Linux
""")

add("rohit_sharma.txt", "txt", """
Rohit Sharma
Noida, India | rohit.sharma@email.com | +91 98110 99887

SUMMARY
Backend engineer with 7 years of experience in enterprise Java systems.

EXPERIENCE

Technical Lead, Aravalli Enterprise Solutions
2021 - Present
- Leads a team of five building Spring Boot microservices for insurance
  claims processing.
- Oracle and Postgres database design; complex SQL and stored procedures.
- Migrated the monolith to Kubernetes; Docker, Jenkins CI/CD.

Senior Software Engineer, Yamuna Infotech
2019 - 2021
- Java, Spring, Hibernate; REST API development; JUnit testing.

EDUCATION
B.Tech in Information Technology, Amity University, 2019

SKILLS
Java, Spring Boot, Hibernate, Maven, REST API, Oracle, Postgres, SQL,
Kubernetes, Docker, Jenkins, Git, Linux, Agile, JIRA
""")


# --------------------------------------------------------------------------
# Writers
# --------------------------------------------------------------------------
def write_txt(path: Path, body: str) -> None:
    path.write_text(body + "\n", encoding="utf-8")


def write_docx(path: Path, body: str) -> None:
    import docx
    document = docx.Document()
    for line in body.split("\n"):
        document.add_paragraph(line)
    document.save(str(path))


def write_pdf(path: Path, body: str) -> None:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(str(path), pagesize=A4)
    width, height = A4
    y = height - 20 * mm
    for line in body.split("\n"):
        if y < 20 * mm:
            c.showPage()
            y = height - 20 * mm
        c.setFont("Helvetica", 9)
        c.drawString(18 * mm, y, line[:105])
        y -= 4.6 * mm
    c.save()


def write_scanned_pdf(path: Path) -> None:
    """An image-only PDF with no text layer.

    Included on purpose: it is the most common real-world parsing failure, and
    the agent must warn about it rather than silently scoring the candidate
    as having zero skills.
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(str(path), pagesize=A4)
    # Grey rectangles standing in for a scanned page image. No text operators
    # are emitted, so pypdf finds nothing to extract.
    c.setFillColorRGB(0.85, 0.85, 0.85)
    for i in range(24):
        c.rect(20 * mm, (250 - i * 9) * mm, (90 + (i * 7) % 70) * mm, 4 * mm,
               stroke=0, fill=1)
    c.save()


WRITERS = {"txt": write_txt, "docx": write_docx, "pdf": write_pdf}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for filename, fmt, body in RESUMES:
        WRITERS[fmt](OUT / filename, body)
        print(f"wrote {filename} ({fmt})")
    write_scanned_pdf(OUT / "scanned_unreadable.pdf")
    print("wrote scanned_unreadable.pdf (image-only, no text layer - on purpose)")
    print(f"\n{len(RESUMES) + 1} files in {OUT}")


if __name__ == "__main__":
    main()
