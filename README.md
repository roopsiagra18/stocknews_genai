🧠 RockyBot: News Research Tool 📈
RockyBot is an intelligent research assistant that helps users analyze and query news articles using state-of-the-art language models. It fetches content from provided URLs, breaks down the text into chunks, generates semantic embeddings, and allows users to ask questions about the content—powered by Google's Gemini and HuggingFace embeddings.

🚀 Features
📑 Load and process up to 3 news article URLs.

✂️ Intelligent text chunking using RecursiveCharacterTextSplitter.

🧠 Embedding generation using HuggingFaceEmbeddings.

🧲 Semantic search via FAISS vector store.

💬 Conversational QA powered by Gemini (gemini-2.0-flash).

🧾 Source citations for retrieved answers.

🖥️ Simple and interactive UI using Streamlit.

📦 Tech Stack
Component	Technology
Language Model	Google Gemini (langchain_google_genai)
Embeddings	HuggingFace (sentence-transformers/all-MiniLM-L6-v2)
Vector Store	FAISS
Web App UI	Streamlit
Document Loader	Langchain UnstructuredURLLoader
Environment Handling	python-dotenv

🛠️ Setup Instructions
1. Clone the repository
bash
Copy
Edit
git clone https://github.com/your-username/rockybot-news-research.git
cd rockybot-news-research
2. Install dependencies
Ensure you have Python 3.8+ and install the required packages:

bash
Copy
Edit
pip install -r requirements.txt
Sample requirements.txt:

txt
Copy
Edit
streamlit
langchain
langchain-google-genai
sentence-transformers
faiss-cpu
python-dotenv
3. Setup Environment Variables
Create a .env file in the root directory:

env
Copy
Edit
GOOGLE_API_KEY=your_google_gemini_api_key_here
▶️ Run the Application
bash
Copy
Edit
streamlit run app.py
Once the app launches:

Enter up to 3 news article URLs in the sidebar.

Click on "Process URLs" to fetch and embed the articles.

Ask any question related to the content in the input box.

View the answer and its sources.

📁 File Structure
bash
Copy
Edit
rockybot-news-research/
│
├── app.py                # Main Streamlit application
├── .env                  # Environment variables (not committed)
├── requirements.txt      # Python dependencies
├── vector_index.pkl      # Stored FAISS vector index (auto-generated)
🔍 Example Use Case
Input URLs:

https://www.example-news.com/article1

https://www.example-news.com/article2

Query:

"What are the key economic takeaways from these articles?"

Output:

Answer summary from the articles.

Cited source URLs used in generating the answer.

📌 Notes
Ensure all input URLs are valid and accessible.

Gemini API key is required to use Google's ChatGoogleGenerativeAI.

Re-processing the same URLs overwrites the existing FAISS vector store.

🧑‍💻 Author
Roopsi Agrawal
Software Engineer | Backend Developer | AI/ML Enthusiast | Data Engineer
[LinkedIn](https://www.linkedin.com/in/roopsiagrawal/)

