# KIITGPT Chatbot - Fine-tuned Llama 2 with LoRA

This repository contains the code for KIITGPT, a chatbot designed to answer questions about Kalinga Institute of Industrial Technology (KIIT). It utilizes the Llama 2 7B chat model, fine-tuned using LoRA (Low-Rank Adaptation) on a custom KIIT-specific dataset. The chatbot interface is built with Gradio and can be run directly via the included Colab notebook.

**Goal:** To create a helpful and informative AI assistant knowledgeable specifically about KIIT University.

<!-- Optional: Add a badge or link to a live demo if you deploy one later -->
<!-- [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](YOUR_SPACE_LINK_HERE) -->

## Overview

![Screenshot 2025-04-18 170028](https://github.com/user-attachments/assets/3a071d99-8405-47f6-8fab-0c6e07d3bd50)

This project demonstrates the process of fine-tuning a large language model (LLM) for a specific domain using Parameter-Efficient Fine-Tuning (PEFT). We leverage **LoRA (Low-Rank Adaptation)** to adapt the `meta-llama/Llama-2-7b-chat-hf` model without requiring prohibitive amounts of computational resources. The model is loaded with 4-bit quantization (`bitsandbytes`) during inference for efficiency. The fine-tuned adapters capture KIIT-specific knowledge, enabling the chatbot to provide relevant answers via a Gradio interface.

## Key Features & Technologies

*   **Base Model:** `meta-llama/Llama-2-7b-chat-hf` (loaded quantized in 4-bit)
*   **Fine-tuning Technique:** **LoRA** (Low-Rank Adaptation) using the `peft` library.
*   **Quantization:** 4-bit inference via `bitsandbytes`.
*   **Core Libraries:** `transformers`, `peft`, `accelerate`, `bitsandbytes`, `torch`.
*   **User Interface:** Gradio (`gradio`).
*   **Adapter Hosting:** Fine-tuned LoRA adapters hosted on [Hugging Face Model Hub](https://huggingface.co/sujoy0011/kiit-llama2-lora-adapters).
*   **Execution:** Designed primarily for Google Colab (using `app.ipynb`) with GPU acceleration.

## How it Works

1.  **Fine-tuning (LoRA):** *(Note: Fine-tuning code is not in this repo)* The base `meta-llama/Llama-2-7b-chat-hf` model was fine-tuned by inserting small, trainable LoRA adapter layers into its structure. Only these adapters were trained on a custom dataset containing questions and answers about KIIT, significantly reducing the compute required compared to full fine-tuning.
2.  **Adapter Hosting:** The trained LoRA adapter weights are stored on the Hugging Face Hub: [sujoy0011/kiit-llama2-lora-adapters](https://huggingface.co/sujoy0011/kiit-llama2-lora-adapters).
3.  **Inference (Gradio App):**
    *   The application (`app.ipynb` or `app_gradio.py`) loads the base `meta-llama/Llama-2-7b-chat-hf` model, applying 4-bit quantization on-the-fly using `bitsandbytes`.
    *   It then downloads and merges the fine-tuned LoRA adapters from the Hugging Face Hub repository using the `peft` library.
    *   A `transformers` pipeline is created using the combined model (quantized base + adapters).
    *   The Gradio interface captures user questions, formats them into the Llama 2 chat prompt structure, passes them to the pipeline for generation, and displays the response.

## Dataset

The model was fine-tuned on a custom dataset consisting of question-answer pairs specifically related to Kalinga Institute of Industrial Technology (KIIT).

## Setup and Running the Chatbot

### Using Google Colab (Recommended Method)

This is the simplest way to run the chatbot using the provided notebook and free GPU resources.

1.  **Open in Colab:** Upload the `app.ipynb` notebook from this repository to your Google Drive and open it in Google Colab, or open it directly using the Colab link if provided.
2.  **Select GPU Runtime:** In Colab, go to `Runtime` -> `Change runtime type` and select a `GPU` accelerator (e.g., T4).
3.  **Set HF_TOKEN Secret:**
    *   Click the "Key" icon (🔑) in the left sidebar.
    *   Click "Add a new secret".
    *   Name: `HF_TOKEN`
    *   Value: Your Hugging Face access token (get one [here](https://huggingface.co/settings/tokens), needs 'read' access).
    *   Ensure "Notebook access" is ON.
4.  **Accept Llama 2 Terms:** Make sure you have accepted the terms for `meta-llama/Llama-2-7b-chat-hf` on its [Hugging Face model card](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf).
5.  **Run All Cells:** Execute the cells in the `app.ipynb` notebook sequentially. The notebook will:
    *   Install necessary libraries.
    *   Log in to Hugging Face (using your secret).
    *   Load the quantized base model and LoRA adapters.
    *   Launch the Gradio interface.
6.  **Access the App:** Click the public `.gradio.live` URL generated in the output of the last cell to interact with the chatbot.

*(Note: Colab free tier has CPU RAM limits (~12.8GB) which might still be insufficient for loading the 7B model reliably. Colab Pro with a High-RAM runtime may be required for stable operation.)*

### Running Locally (Optional, Requires Setup)

1.  **Clone Repository:**
    ```
    git clone https://github.com/dutta-sujoy/KiitGPT.git
    cd KiitGPT
    ```
2.  **Create Environment & Install:**
    ```
    python -m venv venv
    source venv/bin/activate # Linux/macOS or venv\Scripts\activate for Windows
    pip install -r requirements.txt
    ```
3.  **Set HF_TOKEN:** Create a `.env` file in the root directory:
    ```
    HF_TOKEN="hf_YOUR_HUGGINGFACE_TOKEN"
    ```
    Also accept Llama 2 terms (see step 4 in Colab section).
4.  **GPU Required:** Ensure you have a compatible NVIDIA GPU with appropriate drivers and CUDA toolkit installed.
5.  **Run Script:**
    ```
    python app_gradio.py
    ```
6.  **Access:** Open the local URL (e.g., `http://127.0.0.1:7860`) in your browser.

## Preview



https://github.com/user-attachments/assets/98134ad5-8f24-485a-a73b-4d9619cba117




## Challenges & Limitations

*   **Resource Requirements:** Requires a GPU with sufficient VRAM (~10-15GB minimum) and adequate CPU RAM (~16GB+ recommended) for stable loading and inference.
*   **Colab Limits:** Free Colab sessions have time/resource limitations.
*   **Knowledge Cutoff:** The chatbot's knowledge is limited to the information present in its base model training and the custom KIIT fine-tuning dataset. It may not have the latest information.
*   **Gradio Share Link Expiry:** When running on Colab, the public `.gradio.live` link expires after 72 hours or when the session ends.

## Acknowledgements

*   **Meta AI** for the Llama 2 base model.
*   **Hugging Face** for the `transformers`, `peft`, `accelerate`, `bitsandbytes`, `datasets`, and `gradio` libraries and the platform.
*   **Tim Dettmers et al.** for the `bitsandbytes` library.
*   The **LoRA** authors for the parameter-efficient fine-tuning technique.

---

<div style="text-align: center; margin-top: 20px; font-size: small; color: #888;">
    Made with ❤️ by <a href="https://www.linkedin.com/in/dutta-sujoy/" target="_blank" style="color: #007bff; text-decoration: none;">Sujoy</a>
</div>


