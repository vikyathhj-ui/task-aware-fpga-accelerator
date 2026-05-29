\#Design of a Systolic Array-Based FPGA Accelerator for Task-Aware Object Detection on Edge Devices


A research-oriented multimodal affordance reasoning pipeline designed for intelligent task-aware object selection on edge devices. Unlike conventional object detection systems, the proposed architecture performs semantic reasoning to determine the most suitable object for a given task using contextual affordance scoring, transformer-based embeddings, and FPGA-aware systolic similarity computation.



\## Architecture Overview



The pipeline consists of a 9-stage embedded AI workflow:



1\. Image Input

2\. YOLO / YOLO-World Object Detection

3\. Transformer-Based Semantic Embedding Generation

4\. Task \& Prior Retrieval

5\. Affordance and Semantic Scoring

6\. Contextual Reasoning

7\. Semantic Rejection Gating

8\. Object Selection

9\. Output Visualization \& Latency Profiling



The system integrates:



\* YOLOv8 / YOLO-World for object detection

\* SentenceTransformers / CLIP embeddings for semantic reasoning

\* Context-aware affordance scoring

\* FPGA-aware systolic similarity simulation

\* Semantic rejection gating

\* Embedded CPU-compatible deployment



\## Implemented Tasks



1\. Step on something

2\. Sit comfortably

3\. Place flowers

4\. Get potatoes out of fire

5\. Water a plant

6\. Get lemon out of tea

7\. Dig a hole

8\. Open a bottle of beer

9\. Open a parcel

10\. Serve wine

11\. Pour sugar

12\. Smear butter

13\. Extinguish fire

14\. Pound carpet



\## Technologies Used



\* Python

\* PyTorch

\* YOLOv8 / YOLO-World

\* SentenceTransformers

\* CLIP

\* OpenCV

\* NumPy



\## Run Procedure



Install dependencies:



```bash

pip install -r requirements.txt

```



Run inference:



```bash

python pipeline.py image.jpg --task 9

```



Enable CLIP semantic scoring:



```bash

python pipeline.py image.jpg --task 9 --clip

```



Use YOLO-World backend:



```bash

python pipeline.py image.jpg --task 9 --backend yolow

```



List all supported tasks:



```bash

python pipeline.py image.jpg --task 1 --tasks

```



\## Experimental Outcomes



\* Overall task-selection accuracy: \~84%

\* Rejection accuracy on unsuitable scenes: \~91%

\* End-to-end CPU latency: \~420–580 ms/image

\* FPGA systolic similarity latency estimate: <10 µs



\## Key Contributions



\* Task-aware multimodal affordance reasoning

\* Semantic rejection gating

\* Context-aware object selection

\* FPGA-aware systolic similarity acceleration

\* Transformer-based semantic reasoning pipeline

\* Embedded AI deployment architecture



